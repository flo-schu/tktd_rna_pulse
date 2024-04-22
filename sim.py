import numpy as np
import xarray as xr
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.stats import lognorm, binom
import pandas as pd
import os
import itertools

from pymob.utils.store_file import read_config
from pymob.simulation import SimulationBase
from pymob.utils.store_file import prepare_casestudy
from pymob.sim.base import stack_variables
from pymob.utils.config import lambdify_expression, lookup_args

import mod
import prob
import data
import plot

from mod import solver, solve, tktd_rna_3, tktd_guts_minimal
from data import load_data, query_database_single_substance

def xarray_indexer(ds, indices: dict, original_index="id"):
    for new_index, value in indices.items():
        ds_swapped = ds.swap_dims({original_index: new_index})
        idx = ds_swapped.sel({new_index: value}).id
        ds = ds.sel({original_index: np.array(idx.values, ndmin=1)})

    return ds
    
def is_iterable(x):
    try:
        iter(x)
        return True
    except TypeError:
        return False
    
class Simulation(SimulationBase):
    __pymob_version__ = "0.3.0a3"
    solver = solve
    model = tktd_rna_3
    mod = mod
    prob = prob
    dat = data

    @staticmethod
    def parameterize(free_parameters, model_parameters):
        
        # substance = self.config.get("simulation", "substance")
        y0 = model_parameters["y0"]
        parameters = model_parameters["parameters"] 
        parameters.update(free_parameters)

        # here I could use a prior, which varies cext_0_nom by a standard
        # deviation

        params = {"y0": y0, "parameters": parameters}
        return params

    def generate_artificial_data(self, nan_frac=0.2):  
        # create artificial data from Evaluator      
        evaluator = self.dispatch(self.model_parameter_dict)
        evaluator()
        y = evaluator.results

        nzfe = self.observations.nzfe

        # add noise
        rng = np.random.default_rng(seed=1)
        EPS = 1e-8
        y["cext"].values = rng.lognormal(mean=np.log(y["cext"].values + EPS), sigma=0.1)
        y["cint"].values = rng.lognormal(mean=np.log(y["cint"].values + EPS), sigma=0.1)
        y["nrf2"].values = rng.lognormal(mean=np.log(y["nrf2"].values + EPS), sigma=0.1)
        y["lethality"].values = rng.binomial(n=nzfe, p=y["lethality"].values)

        # add missing data
        for i, (k, val) in enumerate(y.items()):
            nans = rng.binomial(n=1, p=nan_frac, size=val.shape)
            val.values = np.where(nans == 1, np.nan, val.values)
            y.update({k: val})

        # this is added last so that NZFE does not contain missing data
        y["nzfe"] = nzfe
        return y

    def initialize(self, input):
        self.load_functions()
        observations = self.load_observations()
        observations = self.reshape_observations(observations)
        observations = self.postprocess_observations(observations)
        self.observations = observations
        self.set_fixed_parameters(input=input)
        self.set_y0()

    def load_functions(self):
        _model = self.config["simulation"].get("model", fallback=None)
        if _model is not None:
            self.model = getattr(mod, _model)

        _solver = self.config["simulation"].get("solver", fallback=None)
        if _solver is not None:
            self.solver = getattr(mod, _solver)

    def set_fixed_parameters(self, input):
        model_params = read_config(input[0])["model"]
        model_params["volume_ratio"] = float(model_params["volume_ratio"])
        self.model_parameters["parameters"] = model_params

    def load_observations(self):
        # get options from config
        hpf = self.config.getfloat("simulation", "hpf")
        substance = self.config.getlist("simulation", "substance")
        if isinstance(substance, str):
            substance = [substance]
        ids = self.config.getlist("simulation", "ids", fallback=None)
        exclude_experiments = np.array(self.config.getlist(
            "simulation", "exclude_experiments", fallback=[]
        ), dtype=int)
        include_experiments = np.array(self.config.getlist(
            "simulation", "include_experiments", fallback=999999
        ), dtype=int)
        exclude_treatments = np.array(self.config.getlist(
            "simulation", "exclude_treatments", fallback=[]
        ), dtype=int)
        include_treatments = np.array(self.config.getlist(
            "simulation", "include_treatments", fallback=999999
        ), dtype=int)
        ids = [ids] if isinstance(ids, str) else ids

        observations = []
        for sub in substance:
            srange = f"substance_range_{sub}"
            
            substance_range = self.config.getlistfloat(
                "simulation", srange, 
                fallback=[0, np.inf]
            )
            
            # execute query and return xarray
            obs = query_database_single_substance(
                database="data/tox.db",
                hpf=hpf,
                substance=sub, 
                substance_range=substance_range
            )

            observations.append(obs)

        observations: xr.Dataset = xr.concat(observations, dim="id")

        # filter by id (treamtment+replicate)
        if ids is not None:
            observations = observations.sel(id=ids)

        # include experiment IDs
        if np.all(include_treatments != 999999):
            observations = observations.where(
                observations.experiment_id.isin(include_experiments), 
                drop=True
            )

        # exclude experiment IDs
        observations = observations.where(
            ~observations.experiment_id.isin(exclude_experiments), 
            drop=True
        )

        # include experiment IDs
        if np.all(include_treatments != 999999):
            observations = observations.where(
                observations.treatment_id.isin(include_treatments), 
                drop=True
            )

        # exclude treatment IDs
        observations = observations.where(
            ~observations.treatment_id.isin(exclude_treatments), 
            drop=True
        )

        observations.attrs["substance"] = substance
        observations.attrs["ids_subset"] = ids
        observations.attrs["excluded_experiments"] = exclude_experiments

        return observations

    def reshape_observations(self, observations):
        stacked_obs = stack_variables(
            ds=observations.copy(),
            variables=["cext", "cint", "cext_nom"],
            new_coordinates=observations.attrs["substance"],
            new_dim="substance",
            pattern=lambda var, coord: f"{var}_{coord}"
        )
        return stacked_obs[self.data_variables].transpose(*self.dimensions)

    def postprocess_observations(self, observations):
        # Add EPS to allow sampling 
        for data_var in ["cext", "cint"]:
            observations[data_var] += 1e-8

        # change lethality to percent. Must not be done when inference
        # is performed with binomial error model
        # observations["lethality"].values = (
        #     observations["lethality"] / observations.nzfe)

        # instead check if all datapoints with lethality information have an
        # associated number of ZFE that were used in the mortalitly experiment
        max_leth = observations.lethality.max(dim="time", skipna=True)
        nzfe = observations.nzfe

        # the query makes sets all ids with a non-nan value for nzfe to 1 and
        # all values which have at least one reported lethality value also to 1
        # the difference between both arrays must be greater or erqual to
        # zero to pass the test, because any id with a value of -1 would
        # indiacate that a lethality observation existed where no information about
        # the number of zfes is included. Such information should either be
        # excluded or nzfe information should be looked up in the raw data.
        logical_query = (~nzfe.isnull()).astype(int) - (~max_leth.isnull()).astype(int) 
        ids_with_missing_nzfe = nzfe.id[np.where(logical_query < 0)[0]].values
        assert len(ids_with_missing_nzfe) == 0, (
            f"{ids_with_missing_nzfe} contained no info about n_zfe in lethality experiments")

        # after this assertion, missing nzfe values can safely be replaced
        # with a default value, whose only purpose is to make the likelihood computation
        # work and should have no effect on inference
        nzfe = np.where(np.isnan(nzfe), 9, nzfe)
        n = len(observations["id"])
        nt = len(observations["time"])
        nzfe = np.array(np.broadcast_to(nzfe, (nt, n)).T).astype(int)
        observations["nzfe"] = (("id", "time"), nzfe)

        # calculate survival from lethality
        survival = observations["nzfe"] - observations["lethality"]
        observations["survival"] = survival
        
        # survival data looks fine
        # fig, axes = plt.subplots(2,3)
        # for s, (axt, axl) in zip(observations.attrs["substance"], axes.T):
        #     obs = observations.where(observations.substance == s, drop=True)
        #     [axt.plot(obs.time, obs.survival.sel(id=i), marker="o") for i in obs.id]
        #     [axl.plot(obs.time, obs.lethality.sel(id=i), marker="o") for i in obs.id]


        # fill nan values forward in time with the last observation 
        # until the next observation. Afterwards leading nans are replaced with 
        # nzfe(t_0) (no lethality observed before the first observation)
        nsurv = survival.ffill(dim="time").fillna(observations.nzfe.isel(time=0))
        # just make sure survival is 1 at the beginning of all experiments
        np.testing.assert_allclose(nsurv.isel(time=0) - observations["nzfe"].isel(time=0), 0)
        
        # create a convenience observation survivors before t, which gives the
        # number of living organisms at the end of time interval t-1
        # this is used for calculating conditional survival
        observations = observations.assign_coords({
            "survivors_before_t": (("id", "time"), np.column_stack([
                nsurv.isel(time=0).values, 
                nsurv.isel(time=list(range(0, len(nsurv.time)-1))).values
        ]).astype(int))})

        # caluclate mortality observed in the different time windows
        # nleth = observations["lethality"].ffill(dim="time").fillna(0)
        # L = observations["lethality"]
        # L_diff = np.column_stack([L, observations["nzfe"].isel(time=0)])[:, 1:] - L

        # replace nominal concentrations with maximum measured external concentrations
        # mean leads to worse relationships
        cext_obs = ~observations.cext.isnull().all(dim="time")
        old_cext = observations["cext_nom"]
        new_cext = observations["cext"].max("time")
        observations["cext_nom"] = xr.where(cext_obs, new_cext, old_cext) 

        # if observations.attrs["ids_subset"] is not None:
        #     observations = observations.sel(id=observations.attrs["ids_subset"])
        
        return observations

    def set_y0(self):
        # generate y0
        y0 = self.observations.isel(time=0).drop("lethality")
        y0["cint"].values = np.zeros(shape=y0["cint"].shape)
        y0["cext"] = xr.where(y0["cext"].isnull(), y0["cext_nom"], y0["cext"])
        y0["nrf2"].values = np.zeros(shape=y0["nrf2"].shape)
        
        self.model_parameters["y0"] = y0
        
    def parse_y0(self, y0=None):
        if y0 is None:
            y0 = self.observations.isel(time=0).copy()

        # parse dims and coords
        y0_dims = {k:v for k, v in self.observations.dims.items() if k != "time"}
        y0_coords = {k:v for k, v in self.observations.coords.items() if k in y0_dims}
        
        y0_list = self.config["simulation"].getlist("y0", fallback=[])
        
        y0_dataset = xr.Dataset()
        for y0_expression in y0_list:
            key, expr = y0_expression.split("=")
            
            func, args = lambdify_expression(expr)

            kwargs = lookup_args(args, y0)

            value = func(**kwargs)

            if not isinstance(value, xr.DataArray):
                value.shape != tuple(y0_dims.values())
                value = np.broadcast_to(value, tuple(y0_dims.values()))
                value = xr.DataArray(value, coords=y0_coords)

            else:
                value = xr.DataArray(value.values, coords=y0_coords)

            y0_dataset[key] = value


        return y0_dataset

    def set_coordinates(self, input):
        observations = self.observations
        sample = observations.id.values
        obs_time = observations.time.values
        substance = observations.substance.values

        # t = np.unique(np.concatenate([
        #     np.linspace(obs_time.min(), obs_time.max(), 1000),
        #         obs_time
        #     ]))
        # t.sort()
        return sample, obs_time, substance

    def run(self):
        # old API discouraged to use. Rather use Evaluator API
        y0 = self.model_parameters["y0"]
        model_params = self.model_parameters["parameters"]
        seed = None
        Y = solver(y0, self.coordinates, model_params, seed)
        return Y

    @staticmethod
    def flatten_parameter_dict(model_parameter_dict, exclude_params=[]):
        parameters = model_parameter_dict

        flat_params = {}
        empty_map = {}
        for par, value in parameters.items():
            if par in exclude_params:
                continue
            if is_iterable(value):
                empty_map.update({par: np.full_like(value, fill_value=np.nan)})
                for i, subvalue in enumerate(value):
                    subpar = f"{par}___{i}"
                    flat_params.update({subpar: subvalue})

            else:
                flat_params.update({par: value})

            if par not in empty_map:
                empty_map.update({par: np.nan})


        def reverse_mapper(parameters):
            param_dict = empty_map.copy()

            for subpar, value in parameters.items():
                subpar_list = subpar.split("___")

                if len(subpar_list) > 1:
                    par, par_index = subpar_list
                    param_dict[par][int(par_index)] = value
                elif len(subpar_list) == 1:
                    par, = subpar_list
                    param_dict[par] = value

            return param_dict
        
        return flat_params, reverse_mapper



    @staticmethod
    def indexer(sim, obs, data_var, idx):
        obs_idx = obs[data_var].values[*idx[data_var]]
        sim_idx = sim[data_var].values[*idx[data_var]]
        return obs_idx, sim_idx

    def log_likelihood(self, results, sigma_cint, sigma_nrf2):
        sim = self.results_to_df(results)
        obs = self.observations
        eps = 1e-8
        obs_idx = {k:np.where(~np.isnan(v.values)) for k, v in obs.items()}
        
        cint_obs, cint_sim = self.indexer(sim, obs, "cint", obs_idx)
        nrf2_obs, nrf2_sim = self.indexer(sim, obs, "nrf2", obs_idx)
        leth_obs, leth_sim = self.indexer(sim, obs, "lethality", obs_idx)

        # compute log likelihoods
        # lp_cext = lognorm.logpdf(x=obs["cext"], scale=sim["cext"], s=theta["sigma_cext"])
        lp_cint = lognorm.logpdf(x=cint_obs, scale=cint_sim+eps, s=sigma_cint)
        lp_nrf2 = lognorm.logpdf(x=nrf2_obs, scale=nrf2_sim, s=sigma_nrf2)
        lp_leth = binom.logpmf(k=leth_obs, n=obs["nzfe"].values[*obs_idx["lethality"]], p=leth_sim)
        
        lp_sum = 0
        for lp in [lp_cint, lp_nrf2, lp_leth]:
            lp_sum += lp.sum()

        return "log-likelihood", -lp_sum

    def fix_inf(self, results: xr.Dataset):
        for i, d in enumerate(self.data_variables):
            r = results[d].values
            if np.sum(np.isinf(r)) > 0:
                print("Infinity in results")

            r = np.where(np.isinf(r), self.scaler.data_max_[i] * 1e3, r)
            results[d].values = r
        return results


    def objective_average(self, results):
        sim = self.scale_(self.fix_inf(self.results_to_df(results)))
        obs = self.observations_scaled
        diff = (sim - obs).to_array()
        objectives = diff.mean(dim=("id", "time"))
        return (objectives ** 2).mean()
    
    def multiobjective_average(self, results):
        diff = (self.scale_results(results) - self.observations_scaled).to_array()
        objectives = diff.mean(dim=("id", "time"))
        return objectives ** 2
    
    def scale_survival_data(self, data_variable):
        obs = self.observations.copy()
        obs[data_variable] = obs[data_variable] / obs["nzfe"]
        return obs

    def plot(self, results):
        # fig = plot.plot_multisubstance(sim=self, results=results)
        results = results.assign_coords({"substance": self.observations.substance})
        results.attrs["substance"] = self.observations.attrs["substance"]
        fig = self.plot_simulation_results(results)
        # fig.savefig(f"{self.output_path}/simulation_results.png")
        
        # obj_names, obj_values = self.objective_function(self.Y)

        # for ax, key, val in zip(fig.axes, obj_names, obj_values):
        #     ax.text(1, 1, s=f"{key} = {round(val, 5)}", transform=ax.transAxes, ha="right")
        # fig = plot.plot_func_for_vars(sim=self, results=results, func=plot.timeseries)
        # fig.savefig(f"{self.output_path}/timeseries.png")

    @staticmethod
    def plot_simulation_results(results: xr.Dataset, cmap=None, data_variables=None, substances=None):
        if data_variables is None:
            data_variables = list(results.data_vars.keys())

        if cmap is None:
            cmap = mpl.colormaps["cool"]
        
        if substances is None:
            substances = results.attrs["substance"]
        
        nv = len(data_variables)
        ns = len(substances) 
        fig, axes = plt.subplots(nv, ns, figsize=(2 + ns*3, nv*1 + 2), 
                                 sharex=True, squeeze=False)
        
        for i, v in enumerate(data_variables):
            for j, s in enumerate(substances):
                ax = axes[i, j]
                substance_res = results.where(results.substance==s, drop=True)
                C = np.round(substance_res.cext_nom.values, 1)
                norm = mpl.colors.Normalize(vmin=C.min(), vmax=C.max())
                if i == len(data_variables) - 1: ax.set_xlabel("Time [h]")
                if i == 0: ax.set_title(s.capitalize())
                for ii in substance_res.id:
                    data = substance_res.sel(id=ii)
                    ax.plot(data.time, data[v], color=cmap(norm(float(data.cext_nom))))
                    if j == 0: ax.set_ylabel(v.capitalize()) 

        return fig

    def plot_variable_substance_combi(self, ax, data_variable, substance, prediction="prior_predictions"):
        exposed = self.observations.cext_nom.sel(substance=substance) > 0
        ids_subset = self.observations.id[exposed]
                
        predictions_ =  getattr(self.inferer, f"plot_{prediction}")
        ax = predictions_(
            x_dim="time",
            data_variable=data_variable, 
            ax=ax,
            n=20,
            seed=1,
            subset={"id": ids_subset, "substance": substance}
        )

        return ax


    def prior_predictive_checks(self):
        fig, axes = plt.subplots(4,3, sharex=True, figsize=(10,8))
        for i, d in enumerate(self.data_variables):
            for j, s in enumerate(np.unique(self.observations.substance.values)):
                ax = self.plot_variable_substance_combi(
                    data_variable=d,
                    substance=s,
                    prediction="prior_predictions",
                    ax=axes[i,j]
                )
                if i == 0:
                    ax.set_title(s)
                if i != 3:
                    ax.set_xlabel("")
                if d == "nrf2":
                    ax.set_yscale("log")
        fig.savefig(f"{self.output_path}/ppc.png")


    def experiment_info(self):
        experiments = self.dat.experiment_table("data/tox.db", self.observations)
        treatments = self.dat.treatment_table("data/tox.db", self.observations)

        datasets = pd.merge(
            treatments, experiments, 
            left_on="experiment_id", right_on="id",
            suffixes=("_treatment", "_experiment")
        )

        # count the number of observations (over time or repeats)
        tid = self.observations.treatment_id
        n_tid = np.bincount(tid)
        datasets["n_treatments"] = n_tid[treatments.id]
        datasets["volume"] = None
        datasets["replicates"] = None

        return datasets

    def posterior_predictive_checks(self):
        self.coordinates["time"] = np.linspace(24, 120, 100)
        fig, axes = plt.subplots(4,3, sharex=True, figsize=(10,8))
        for i, d in enumerate(self.data_variables):
            for j, s in enumerate(np.unique(self.observations.substance.values)):
                ax = self.plot_variable_substance_combi(
                    data_variable=d,
                    substance=s,
                    prediction="posterior_predictions",
                    ax=axes[i,j]
                )
                if i == 0:
                    ax.set_title(s)
                if i != 3:
                    ax.set_xlabel("")
        fig.savefig(f"{self.output_path}/posterior_predictive.png")
        self.coordinates["time"] = self.observations.time.values

    def pyabc_posterior_predictions(self):
        plot.pretty_posterior_plot_multisubstance(self)
        # plot.pyabc_predictions(self)

    @staticmethod
    def get_ids(dataset, indices):
        try:
            data = xarray_indexer(
                ds=dataset, 
                indices=indices, 
                original_index="id"
            )
            obs_ids = np.array(data.id.values, ndmin=1)
        except KeyError:
            obs_ids = np.array([], ndmin=1)

        return obs_ids

        # get all observations from the same experiment

    def plot_experiment(
        self, 
        experiment_id, 
        substance, 
        data_var, 
        ax=None, 
        norm=None,
        cmap=mpl.colormaps["cool"],
    ):
        # get metadata
        experiments = self.dat.experiment_table("data/tox.db", self.observations)
        meta = experiments.query(f"id=={experiment_id}").iloc[0]
        
        if ax is None:
            fig, ax = plt.subplots(1,1)
        
        indices = {
            "substance": substance, 
            "experiment_id": experiment_id,
        }

        try:
            data = xarray_indexer(
                ds=self.observations, 
                indices=indices, 
                original_index="id"
            )[data_var]
        except KeyError:
            ax.text(0.5, 0.5, "No\ndata", transform=ax.transAxes, 
                    ha="center", va="center")
            return ax, meta, np.array([], ndmin=0), norm


        # get all observations from the same experiment
        obs_ids = np.array(data.id.values, ndmin=1)
        
        # get a list of symbols for different concentrations                    
        marker = iter(list(mpl.lines.Line2D.markers.keys())[slice(2,-1)])

        if norm is None:
            C = np.round(data.cext_nom.values, 1)
            norm = mpl.colors.Normalize(vmin=C.min(), vmax=C.max())

        for oid in obs_ids:
            o = data.sel(id=oid)
            o = o.dropna(dim="time")

            d = str(meta["date"]).split(" ")[0]
            ep = str(meta["experimentator"])
            n = meta["name"].capitalize()

            # s = str(o.substance.values).capitalize()
            hpf = float(o.hpf.values)
            c = round(float(o.cext_nom.values), 1)

            x = o.time
        
            col = cmap(norm(c))
            lab = f"$C_{{e,0}} = {c}$ ($ID_t = {int(o.treatment_id)}$)"
            if data_var == "lethality" or data_var == "survival":
                y = o.values / o.nzfe
                ax.set_ylim(-0.05,1.05)
                ax.set_ylabel(data_var.capitalize())

            elif data_var == "cext":
                y = o.values
                ax.set_ylabel(f"$C_{{e}}$ [µmol/l]")

            elif data_var == "cint":
                y = o.values
                ax.set_ylabel(f"$C_{{i}}$ [µmol/l]")

            elif data_var == "nrf2":
                y = o.values
                ax.set_ylabel(f"NRF2 [fc]")

            ax.plot(x, y, lw=.3, marker=next(marker), color=col, label=lab)
            ax.spines[["right", "top"]].set_visible(False)
            ax.set_title(f"{n}: ({ep}, {d}), {hpf} hpf, id={experiment_id}")
            ax.set_xlim(-5,120)
            ax.set_xlabel("time [h]")
            ax.legend(title=f"{substance.capitalize()} [µmol/L]")

        return ax, meta, obs_ids, norm

    def plot_cext_cint_relation(self, observations):
        for s in ["diuron", "diclofenac", "naproxen"]:
            cext_nom = observations.cext.where(~observations.cext.isnull(), drop=True).cext_nom
            cext_meas = observations.cext.where(~observations.cext.isnull(), drop=True).max("time")
            cint_meas = observations.cint.where(~observations.cext.isnull(), drop=True).max("time")

            fig, ax = plt.subplots(1,1)
            ax.set_title(s.capitalize())
            ax.scatter(
                cext_nom.where(cext_nom.substance==s, drop=True), 
                cint_meas.where(cint_meas.substance==s, drop=True), 
                alpha=.75, color="tab:blue", label=f"$C_{{e,nom}}$ vs $C_{{i,max}}$")
            ax.scatter(
                cext_meas.where(cext_meas.substance==s, drop=True), 
                cint_meas.where(cint_meas.substance==s, drop=True), 
                alpha=.75, color="tab:orange",
                label=f"$C_{{e,max}}$ vs $C_{{i,max}}$")
            
            eids = cint_meas.where(cint_meas.substance==s, drop=True).experiment_id
            for i, eid in enumerate(eids):
                ax.text(
                    cext_meas.where(cext_meas.substance==s, drop=True)[i],
                    cint_meas.where(cint_meas.substance==s, drop=True)[i],
                    int(eid)
                )
            ax.set_xlabel(f"$C_e$")
            ax.set_ylabel(f"$C_i$")
            ax.legend()

            fig.savefig(f"case_studies/reversible_damage/results/data/rel_ceci_{s}.png")
            plt.close()

    def plot_experiments(self, plot_individual=True):
        obs = self.observations.swap_dims(id="experiment_id")
        experiments = self.dat.experiment_table("data/tox.db", self.observations)
        cmap = mpl.colormaps["cool"]

        # first grouping is by endpoint, because in endpoint experiments,
        # multiple variables may be recorded
        for endpoint, group in experiments.groupby("name"):
            experiment_ids = group.id.values
            
            # gather all observations in one endpoint
            endpoint_ids = np.concatenate(
                [np.array(obs.sel(experiment_id=eid).id.values, ndmin=1) 
                 for eid in experiment_ids]
            )
            endpoint_obs = self.observations.sel(id=endpoint_ids)
            
            # get all substances in the endpoint data
            esubs = np.unique(endpoint_obs.substance)
            # n_per_substance = np.concatenate([np.unique(xarray_indexer(endpoint_obs, {"experiment_id": ei}).substance_index) for ei in experiment_ids])
            # np.bincount(n_per_substance)

            # get all data_variables with observations in the endpoint data
            evars = [k for k, ov in endpoint_obs.data_vars.items() if (~ov.isnull()).sum() > 0]

            # set up the figure with all panels
            rows = list(itertools.product(esubs, evars))
            ncols = len(experiment_ids)
            nrows = len(rows)
            
            if not plot_individual:
                figsize = (5 * ncols, 5 * nrows)
                fig, axes = plt.subplots(
                    nrows, ncols, figsize=figsize, 
                    sharex=True, sharey="row", squeeze=False
                )
            else:
                axes = np.full((nrows, ncols), None)

            # iterate over substances 
            for i, (s, v) in enumerate(rows):

                # extract data from the endpoint-substance combi
                data = xarray_indexer(endpoint_obs, {"substance": s})
                data = self.observations[v].sel(id=data.id.values)
                
                # create a concentration normalizer
                C = np.round(data.cext_nom.values, 1)
                norm = mpl.colors.Normalize(vmin=C.min(), vmax=C.max())
                
                # iterate over experiments
                for j, eid in enumerate(experiment_ids):
                    
                    # get the correct axis (or create a new figure)
                    ax = axes[i, j]
                    
                    ax, meta, _, _ = self.plot_experiment(
                        experiment_id=eid, 
                        substance=s, 
                        data_var=v, 
                        ax=ax, 
                        norm=None if plot_individual else norm,
                        cmap=cmap,
                    )

                    if not ax.has_data() and plot_individual:
                        plt.close()
                        continue

                    folder = f"case_studies/reversible_damage/results/data/{endpoint}".lower()
                    os.makedirs(folder, exist_ok=True)
                    
                    if plot_individual:
                        ep = str(meta['experimentator'])
                        d = str(meta["date"]).split(" ")[0].replace('-', '')
                        file = f"{v}_{eid}_{s}_{ep}_{d}".lower()
                        fig = ax.figure
                        fig.savefig(f"{folder}/{file}.png")
                        plt.close()

            if not plot_individual:
                fig.subplots_adjust(0.02, 0.02, 0.95, 0.95, 0.1, 0.1)
                fig.savefig(f"{folder}_experiments.png")
                plt.close()


class SingleSubstanceSim(Simulation):
    __pymob_version__ = "0.3.0a5"
    def initialize(self, input):
        self.load_functions()

        observations = self.load_observations()
        observations, indices = self.reshape_observations(observations)
        observations = self.postprocess_observations(observations)
        self.observations = observations
        self.indices = indices

        self.set_fixed_parameters(input=input)
        self.set_y0()

    def set_fixed_parameters(self, input):
        model_params = read_config(input[0])["model"]
        model_params["volume_ratio"] = np.array(model_params["volume_ratio"], dtype=float)
        self.model_parameters["parameters"] = model_params

    def reshape_observations(self, observations):
        # TODO: It seems to me that much of this can also be abstracted
        #       as a reduce an index method.
        reduce_dim = "substance"
        
        substances = observations.attrs[reduce_dim]

        stacked_obs = stack_variables(
            ds=observations.copy(),
            variables=["cext_nom", "cext", "cint"],
            new_coordinates=substances,
            new_dim="substance",
            pattern=lambda var, coord: f"{var}_{coord}"
        ).transpose(*self.dimensions, reduce_dim)

        # reduce cext_obs
        cext_nom = stacked_obs["cext_nom"]
        assert np.all(
            (cext_nom == 0).sum(dim=reduce_dim) == len(substances) - 1
        ), "There are mixture treatments in the SingleSubstanceSim."


        # VECTORIZED INDEXING IS THE KEY
        # https://docs.xarray.dev/en/stable/user-guide/indexing.html#vectorized-indexing

        # Defining a function to reduce the "reduce_dim" dimension to length 1
        def index(array, axis=None, **kwargs):
            # Check that there is exactly one non-zero value in 'substance'
            non_zero_count = (array != 0).sum(axis=axis)
            
            if not (non_zero_count == 1).all():
                raise ValueError(f"Invalid '{reduce_dim}' dimension. It should have exactly one non-zero value.")
            
            return np.where(array != 0)[axis]


        # Applying the reduction function using groupby and reduce
        red_data = stacked_obs.cext_nom
        new_dims = [d for d in red_data.dims if d != reduce_dim]

        reduce_dim_idx = red_data\
            .groupby(*new_dims)\
            .reduce(index, dim=reduce_dim)\
            .rename(f"{reduce_dim}_index")
        
        if stacked_obs.dims[reduce_dim] == 1:
            reduce_dim_idx = reduce_dim_idx.squeeze()
        
        reduce_dim_id_mapping = stacked_obs[reduce_dim]\
            .isel({reduce_dim: reduce_dim_idx})\
            .drop(reduce_dim)\
            .rename(f"{reduce_dim}_id_mapping")
        
        reduce_dim_idx = reduce_dim_idx.assign_coords({
            f"{reduce_dim}": reduce_dim_id_mapping
        })

        # this works because XARRAY is amazing :)
        stacked_obs["cext_nom"] = stacked_obs["cext_nom"].sel({reduce_dim: reduce_dim_id_mapping})
        stacked_obs["cext"] = stacked_obs["cext"].sel({reduce_dim: reduce_dim_id_mapping})
        stacked_obs["cint"] = stacked_obs["cint"].sel({reduce_dim: reduce_dim_id_mapping})
        
        # drop old dimension and add dimension as inexed dimension
        # this is necessary, as the reduced dimension needs to disappear from
        # the coordinates.
        stacked_obs = stacked_obs.drop_dims(reduce_dim)
        stacked_obs = stacked_obs.assign_coords({
            f"{reduce_dim}": reduce_dim_id_mapping,
            f"{reduce_dim}_index": reduce_dim_idx,
        })
        
        indices = {
            reduce_dim: reduce_dim_idx
        }

        return stacked_obs, indices

    def set_y0(self):
        super().set_y0()

        y0 = self.model_parameters["y0"]
        y0 = self.parse_y0(y0=y0)

        self.model_parameters["y0"] = y0

    def set_coordinates(self, input):
        sample = self.observations.id.values
        obs_time = self.observations.time.values
        # substance = self.observations.substance.values
        return sample, obs_time

    def plot_variable_substance_combi(self, ax, data_variable, substance, prediction="prior_predictions"):
        exposed = self.observations\
            .swap_dims(id="substance")\
            .sel(substance=substance)
        ids_subset = exposed.id.values

        if prediction == "posterior_predictions":
            preds = self.inferer.posterior_predictions(
                n=self.inferer.n_predictions, 
                seed=self.seed
            )
            mode = "mean+hdi"

        elif prediction == "prior_predictions":
            preds = self.inferer.prior_predictions(
                n=50, # this will plot 50 x all experiments. This is sufficient to get an idea
                seed=self.seed
            ).prior_predictive
            mode = "draws"
        
        if data_variable == "survival" or data_variable == "lethality":
            obs = self.scale_survival_data(data_variable)
        else:
            obs = self.observations

        ax = self.inferer.plot_predictions(
            observations=obs,
            predictions=preds,
            data_variable=data_variable, 
            x_dim="time",
            ax=ax,
            subset={"id": ids_subset},
            mode=mode
        )
        

        return ax

class SingleSubstanceSim2(SingleSubstanceSim):
    def set_fixed_parameters(self, input):
        self.model_parameters["parameters"] = self.fixed_model_parameters

if __name__ == "__main__":
    config = prepare_casestudy(("reversible_damage", "rna_pulse_3_6c_substance_independent_rna_protein_module"), "settings.cfg")
    
    sim = SingleSubstanceSim2(config=config)
    sim.plot_experiments(plot_individual=False)
    # sim.plot_cext_cint_relation(sim.observations)

    # sim.experiment_info()
    # evaluator = sim.dispatch(theta=sim.model_parameter_dict)
    # evaluator()
    # evaluator.results

    # print(make_jaxpr(evaluator)())

    
    # sim.benchmark(n=100)

    sim.set_inferer(backend="numpyro")
    # sim.config.set("inference.numpyro", "kernel", "map")
    # sim.config.set("inference.numpyro", "svi_learning_rate", "0.001")
    # sim.config.set("inference.numpyro", "svi_iterations", "500")

    # sim.prior_predictive_checks()
    # sim.inferer.run()
    # sim.inferer.store_results()
    sim.inferer.load_results()
    # sim.posterior_predictive_checks()
    plot.pretty_posterior_plot_multisubstance(sim)
    sim.inferer.plot()
    # sim.inferer.combine_chains(chain_location="chains_nuts", drop_extra_vars=["lethality", "D"])


    # data_vars = ["cext", "cint", "nrf2", "lethality"]
    # fig, axes = plt.subplots(len(data_vars), 3, figsize=(8,12), sharex=True)
    # for i, v in enumerate(data_vars):
    #     for j, s in enumerate(sim.observations.attrs["substance"]):
    #         ids = sim.observations.swap_dims(id="substance").sel(substance=s).id.values
    #         sim.inferer.idata.posterior_predictive[v]\
    #             .sel(id=ids)\
    #             .mean(("chain", "draw"))\
    #             .plot.line(x="time", ax=axes[i,j])
    #         sim.observations[v]\
    #             .sel(id=ids)\
    #             .plot.scatter(x="time", ax=axes[i,j])
            
    # fig.savefig(f"{sim.output_path}/complete_predictions.png")
