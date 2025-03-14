import numpy as np
import xarray as xr
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
import os
import itertools

from pymob.utils.store_file import read_config
from pymob.simulation import SimulationBase
from pymob.utils.store_file import prepare_casestudy
from pymob.sim.base import stack_variables, unlist_attrs, enlist_attr
from pymob.utils.config import lambdify_expression, lookup_args
from pymob.solvers import JaxSolver

from . import mod
from . import prob
from . import data
from . import plot

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
    __pymob_version__ = "0.4.1"

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
        
        # binomial error model
        nzfe_ = np.broadcast_to(nzfe.values.reshape((-1, 1)), y["lethality"].values.shape)
        y["lethality"].values = rng.binomial(n=nzfe_, p=y["lethality"].values)

        # add missing data
        for i, (k, val) in enumerate(y.items()):
            nans = rng.binomial(n=1, p=nan_frac, size=val.shape)
            val.values = np.where(nans == 1, np.nan, val.values)
            y.update({k: val})

        # this is added last so that NZFE does not contain missing data
        y = y.assign_coords({"nzfe": nzfe})
        return y

    def initialize(self, input):
        observations = self.load_observations()
        observations = self.reshape_observations(observations)
        observations = self.postprocess_observations(observations)
        self.observations = observations
        self.set_y0()

    def load_observations(self):
        # get options from config
        hpf = float(self.config.simulation.hpf) # type: ignore
        substance = self.config.simulation.substance.split(" ") # type: ignore
        if isinstance(substance, str):
            substance = [substance]
        ids = self.config.simulation.model_extra.get("ids", None) # type: ignore
        exclude_experiments = np.array(
            self.config.simulation.model_extra.get("exclude_experiments", "999999").split(" "), # type: ignore
            dtype=int
        )
        include_experiments = np.array(
            self.config.simulation.model_extra.get("include_experiments", "999999").split(" "), # type: ignore
            dtype=int
        )
        exclude_treatments = np.array(
            self.config.simulation.model_extra.get("exclude_treatments", "999999").split(" "), # type: ignore
            dtype=int
        )
        include_treatments = np.array(
            self.config.simulation.model_extra.get("include_treatments", "999999").split(" "), # type: ignore
            dtype=int
        )
        ids = [ids] if isinstance(ids, str) else ids

        observations = [] # type: ignore
        for sub in substance:
            srange = f"substance_range_{sub}"
            
            substance_range = self.config.simulation.model_extra.get( # type: ignore
                srange, [0, np.inf]
            ) 

            # execute query and return xarray
            obs = data.query_database_single_substance( # type: ignore
                database=f"{self.data_path}/tox.db",
                hpf=hpf,
                substance=sub, 
                substance_range=substance_range
            )

            observations.append(obs)

        observations: xr.Dataset = xr.concat(observations, dim="id") # type: ignore

        # filter by id (treamtment+replicate)
        if ids is not None:
            observations = observations.sel(id=ids)
        else:
            ids = []

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
        observations = self.lethality_to_conditional_survival(observations)

        # caluclate mortality observed in the different time windows
        # nleth = observations["lethality"].ffill(dim="time").fillna(0)
        # L = observations["lethality"]
        # L_diff = np.column_stack([L, observations["nzfe"].isel(time=0)])[:, 1:] - L

        # replace nominal concentrations with maximum measured external concentrations
        # mean leads to worse relationships
        if not hasattr(self.config.simulation, "use_nominal_concentrations"):
            self.config.simulation.use_nominal_concentrations = False
        
        if not bool(self.config.simulation.use_nominal_concentrations):
            cext_obs = ~observations.cext.isnull().all(dim="time")
            old_cext = observations["cext_nom"]
            new_cext = observations["cext"].max("time")
            observations["cext_nom"] = xr.where(cext_obs, new_cext, old_cext) 


        # if observations.attrs["ids_subset"] is not None:
        #     observations = observations.sel(id=observations.attrs["ids_subset"])
        
        return observations

    @staticmethod
    def lethality_to_conditional_survival(observations):
        nzfe = observations["nzfe"]
        nzfe = np.where(np.isnan(nzfe), 9, nzfe)
        n = len(observations["id"])
        nt = len(observations["time"])
        nzfe = np.array(np.broadcast_to(nzfe, (nt, n)).T).astype(int)
        observations["nzfe"] = (("id", "time"), nzfe)

        # calculate survival from lethality
        survival = observations["nzfe"] - observations["lethality"]
        observations["survival"] = survival
        
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

        return observations
    
    def set_y0(self):
        # generate y0
        y0 = self.observations.isel(time=0).drop("lethality")
        y0["cint"].values = np.zeros(shape=y0["cint"].shape)
        y0["cext"] = xr.where(y0["cext"].isnull(), y0["cext_nom"], y0["cext"])
        y0["nrf2"].values = np.zeros(shape=y0["nrf2"].shape)
        
        self.model_parameters["y0"] = y0

    @staticmethod
    def indexer(sim, obs, data_var, idx):
        obs_idx = obs[data_var].values[*idx[data_var]] # noqa: E999
        sim_idx = sim[data_var].values[*idx[data_var]] # noqa: E999
        return obs_idx, sim_idx

    def objective_average(self, results):
        sim = self.scale_(self.results_to_df(results))
        obs = self.observations_scaled
        diff = (sim - obs).to_array()
        objectives = diff.mean(dim=("id", "time"))
        return (objectives ** 2).mean()
    
    def scale_survival_data(self, data_variable):
        obs = self.observations.copy()
        obs[data_variable] = obs[data_variable] / obs["nzfe"]
        return obs

    def plot(self, results):
        results = results.assign_coords({"substance": self.observations.substance})
        results.attrs["substance"] = self.observations.attrs["substance"]
        fig = self._plot.plot_simulation_results(results)

    def prior_predictive_checks(self):
        fig, axes = plt.subplots(len(self.config.data_structure.observed_data_variables),3, sharex=True, figsize=(10,8))
        for i, d in enumerate(self.config.data_structure.observed_data_variables):
            for j, s in enumerate(np.unique(self.observations.substance.values)):
                ax = self._plot.plot_variable_substance_combi(
                    self,
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
        experiments = self._data.experiment_table("data/tox.db", self.observations)
        treatments = self._data.treatment_table("data/tox.db", self.observations)

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
        fig, axes = plt.subplots(len(self.config.data_structure.observed_data_variables),3, sharex=True, figsize=(10,8))
        for i, d in enumerate(self.config.data_structure.observed_data_variables):
            for j, s in enumerate(np.unique(self.observations.substance.values)):
                ax = self._plot.plot_variable_substance_combi(
                    self,
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
        self._plot.pretty_posterior_plot_multisubstance(self)

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



class SingleSubstanceSim(Simulation):
    __pymob_version__ = "0.4.1"
    def initialize(self, input):

        observations = self.load_observations()
        observations, indices = self.reshape_observations(observations)

        # export observations
        obs_export = unlist_attrs(observations)
        obs_export.to_dataframe().reset_index()[[
            "id", "treatment_id", "experiment_id", "nzfe", "hpf", "substance", 
            "cext_nom", "time", "cext", "cint", "nrf2", 
            "lethality"
        ]].to_csv(f"{self.data_path}/dataset.csv")
        observations = enlist_attr(observations, "substance")
        
        observations = self.postprocess_observations(observations)
        self.observations = observations  #type:ignore
        self.indices = indices

        self.set_y0()

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
            
            return np.where(array != 0)[axis]  #type: ignore


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
            .drop(reduce_dim).rename(f"{reduce_dim}_id_mapping")  # type: ignore
        
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
        y0 = self.parse_input(input="y0", reference_data=y0, drop_dims=["time"])

        self.model_parameters["y0"] = y0

    def set_coordinates(self, input):
        sample = self.observations.id.values
        obs_time = self.observations.time.values
        # substance = self.observations.substance.values
        return sample, obs_time


class SingleSubstanceSim2(SingleSubstanceSim):
    def set_fixed_parameters(self, input):
        self.model_parameters["parameters"] = self.config.model_parameters.value_dict


class SingleSubstanceSim3(SingleSubstanceSim2):
    solver = JaxSolver
    _plot = plot

    def initialize(self, input):
        super().initialize(input)
        self.config.simulation.batch_dimension = "id"
        self.model_parameters["parameters"] = self.config.model_parameters.value_dict

    def prior_predictive_checks(self):
        SimulationBase.prior_predictive_checks(self)

    def posterior_predictive_checks(self):
        SimulationBase.posterior_predictive_checks(self)

        self._plot.pretty_posterior_plot_multisubstance(self)
    

if __name__ == "__main__":
    config = prepare_casestudy((
        "tktd-rna-pulse", 
        "rna_pulse_3_6c_substance_independent_rna_protein_module"), 
        "settings.cfg"
    )
    
    sim = SingleSubstanceSim2(config=config)

    sim._plot.plot_experiments(sim)

    # run a single simulation
    evaluator = sim.dispatch(theta=sim.model_parameter_dict)
    evaluator()
    evaluator.results

    sim.set_inferer(backend="numpyro")

    # run inference
    sim.prior_predictive_checks()
    sim.inferer.run()
    sim.inferer.store_results()  # type: ignore

    # plot inference results
    sim.inferer.load_results()
    sim.posterior_predictive_checks()
    sim.inferer.plot()  # type: ignore
