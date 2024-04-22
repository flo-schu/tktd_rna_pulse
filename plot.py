import os
import numpy as np
import matplotlib as mpl
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex, Normalize
import arviz as az

from pymob.simulation import SimulationBase
from toopy.plot import letterer, draw_axis_letter

def timeseries(results, observations, variable: str, ax=None, title=""):
    if ax is None:
        ax = plt.subplot(111)
    
    results = results[variable]
    observations = observations[variable]

    # consider reverting this change and iterate over ids in 
    # plot method of Simulation
    # plot simulation results and experimental observations
    n = max(len(observations.dropna(dim="id", how="all").id), 1)
    a = 0.2 + 0.8 / n

    for i in observations.id:
        obs = observations.sel(id=i)
        res = results.sel(id=i)
        if np.all(obs.isnull().all().data):
            continue
        ax.plot(res.time, res, ls="-", marker="", color="black", alpha=a)
        ax.plot(obs.time, obs, ls="", marker="o", color="grey", alpha=a)

    # decorate axis
    ax.set_ylabel(variable)
    ax.set_xlabel("Time [h]")
    ax.set_title(title)

    return ax

def plot_predictions(
        results, observations, 
        data_variable: str, x_dim: str, ax=None, subset={}, 
        pred_col="black", obs_col="tab:blue"
    ):
    obs = observations.sel(subset)
    preds = results.sel(subset)
    
    if ax is None:
        ax = plt.subplot(111)
    
    ax.plot(
        preds[x_dim].values, preds[data_variable].values.T, 
        color=pred_col, lw=.8, alpha=.25
    )

    ax.plot(
        obs[x_dim].values, obs[data_variable].values.T, 
        marker="o", ls="", ms=3, color=obs_col
    )
    
    ax.set_ylabel(data_variable)
    ax.set_xlabel(x_dim)

    return ax

def plot_multisubstance(sim, results, column="substance", 
                        color_dict={"naproxen": "tab:red", "diuron": "tab:blue", "diclofenac": "tab:green"}):
    R = len(sim.data_variables)

    fig = plt.figure()
    gs = fig.add_gridspec(nrows=R, ncols=1)

    for i, v in enumerate(sim.data_variables):
        v_coords = results[v].coords
        col_coords = v_coords.get(column, [None])
        gs_sub = gs[i].subgridspec(nrows=1, ncols=len(col_coords))
        axes = gs_sub.subplots(sharex=True)


        if len(col_coords) == 1:
            axes = [axes]

        for cc, ax in zip(col_coords, axes):
            ax = plot_predictions(
                results, 
                sim.observations,
                data_variable=v,
                x_dim="time",
                ax=ax,
                subset={} if cc is None else {column: cc}
            )


    fig.subplots_adjust(right=.98, bottom=0.07, top=1, left=.09, wspace=.28, hspace=.07)
    return fig

def plot_func_for_vars(sim, results, func):
    """Plot a function over all available data variables with shared X
    Removes duplicate axis titles and x-labels and saves the figure
    
    Arguments
    ---------

    func: [callable] a function that takes the arguments sim[SimulationBase],
        variable[str] and ax[Axis] and returns an Axis object
    rel_path: the relative path to the output directory.

    """
    variables = sim.data_variables
    title = sim.config.get("simulation", "substance")
    fig, axes = plt.subplots(len(variables), 1, sharex=True)
    
    for i, (ax, v) in enumerate(zip(axes, variables)):
        ax = func(
            results=results, 
            observations=sim.observations, 
            variable=v, 
            ax=ax,
            title=title
        )
        if i < len(variables) - 1:
            ax.set_xlabel("")
        if i != 0:
            ax.set_title("")

    return fig

def cice_relation(sim: SimulationBase):
    ci = sim.observations.cint_diuron.dropna("id", "all").max("time")
    ce = sim.observations.cext_diuron.dropna("id", "all").isel(time=0)

    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.scatter(ci.cext_nom_diuron, ci.values)
    ax2.scatter(ce.cext_nom_diuron, ce.values)
    ax2.set_xlabel(r"$C_e$ nominal")
    ax2.set_ylabel(r"$C_e$ at $t_0$")
    ax1.set_ylabel(r"$C_i$ max")
    fig.savefig(f"{sim.output_path}/ci_ce_relationship.png")
    

def step(sim):
    # plot the stable step function from rna_pulse_simplified model
    stable_step = lambda Ci, z_ci, v_rt: 0.5 + (1 / np.pi) * np.arctan(v_rt * (Ci - z_ci))
    fig, ax = plt.subplots(1,1)

    x = np.linspace(0,1000,1000)
    ax.plot(x, stable_step(x, 500, 10), label="k=10")
    ax.plot(x, stable_step(x, 500, 1), label="k=1")
    ax.plot(x, stable_step(x, 500, 0.1), label="k=0.1")
    ax.plot(x, stable_step(x, 500, 0.01), label="k=0.01")
    ax.legend()

    fig.savefig(f"{sim.output_path}/step_func.png")

def pyabc_predictions(sim):
    sim.inferer.load_results()
    dp = sim.output_path + f"/predictions_{sim.inferer.history.id}"
    os.makedirs(dp, exist_ok=True)
    
    fig_chains = sim.inferer.plot_chains()
    fig_chains.savefig(f"{sim.output_path}/chains.png")

    coords = sim.coordinates.copy()
    for j, iid in enumerate(coords["id"]):
        print(f"Making posterior prediction ({j+1}/{len(coords['id'])})", end="\r")
        sim.coordinates["id"] = coords["id"][j:j+1]
        sim.coordinates["time"] = np.linspace(24,120,200)

        fig, axes = plt.subplots(4,1, sharex=True)
        for i, (ax, dv) in enumerate(zip(axes, sim.data_variables)):
            ax = sim.inferer.plot_posterior_predictions(
                data_variable=dv, 
                x_dim="time",
                subset={"id": iid},
                ax=ax
            )
            if i != 3: ax.set_xlabel("")
        
        fig.savefig(dp + f"/pyabc_posterior_predictions_dbid{iid}.png")
        plt.close()

    sim.coordinates = coords


def pretty_posterior_plot_multisubstance(sim):
    print("PRETTY PLOT: starting...")

    substances = sim.observations.attrs["substance"]
    # sim.inferer.load_results()
    old_time = sim.coordinates["time"].copy()
    sim.coordinates["time"] = np.linspace(24,120,200)
    x_dim="time"
    obs_raw = sim.observations
    obs_raw.survival.values = (obs_raw.survival / obs_raw.nzfe).values

    for s in substances:
        susbtance_mask = sim.observations.substance == s
        obs = obs_raw.where(susbtance_mask, drop=True)
        cext_0 = sim.model_parameters["y0"][f"cext"].where(susbtance_mask, drop=True)
        if s == "naproxen":
            bins = [100, 200, 300, 400, 1000, 1500]
            midpoint = 400
        elif s == "diclofenac":
            bins = [6,8,20,100]
            midpoint = 13
        elif s == "diuron":
            bins = [5, 12, 24, 48, 100]
            midpoint = 40
        digitized = np.digitize(cext_0, bins)
        
        obs = obs.assign_coords(cext_group=("id", digitized)).swap_dims(id="cext_group")
        data_variables = [v for v in  sim.data_variables if "cext" not in v]

        fig, axes = plt.subplots(nrows=len(data_variables), ncols=len(bins), 
                                sharex=True, sharey="row", figsize=(11,7), squeeze=False)

        panelletter = letterer("abcdefghijklmnopqrstuvwxyz")

        # colormaps = ["Greens", "GnBu", "Blues", "Oranges"]

        pretty_labels = {
            f"cext": f"$C_{{ext}}$ {s.capitalize()}",
            f"cint": f"$C_{{int}}$ {s.capitalize()}",
            f"nrf2": "Fold-change NRF2",
            f"survival": "Survival (ratio)"
        }

        limits = {
            "nrf2": {"diuron": 3, "diclofenac": 5, "naproxen": 5},
        }

        post_pred = sim.inferer.posterior_predictions(
            n=sim.inferer.n_predictions, 
            seed=sim.seed
        )

        cext0 = post_pred[f"cext"].where(post_pred[f"cext"] < bins[-1], drop=True).mean(("draw", "chain"))
        cmap = mpl.colormaps["cool"]
        cmax = float(cext0.max())
        cmin = float(cext0.min())
        # minmax_scaler = lambda x: (x - cmin) / (cmax - cmin)
        minmax_scaler = mpl.colors.TwoSlopeNorm(vmin=cmin, vcenter=midpoint, vmax=cmax)
        
        for bi, bin in enumerate(bins):
            print(f"PRETTY PLOT: make predictions for {s.capitalize()} in bin ({bi+1}/{len(bins)})")
            obs_bin = obs.sel(cext_group=bi)
            sample_ids = obs_bin.id.values
            cext0_bin = post_pred.sel(id=sample_ids)[f"cext"].isel(time=0).mean(("draw", "chain"))
            # cmap = cm.get_cmap("plasma")
            # cmap = mpl.colormaps["cool"]
            # cmax = float(cext0_bin.max())
            # cmin = float(cext0_bin.min())
            # minmax_scaler = lambda x: (x - cmin) / (cmax - cmin)
            for si, sample in enumerate(sample_ids):
                pp_s = post_pred.sel(id=sample)
                ob_s = obs_bin.swap_dims(cext_group="id").sel(id=sample)
                cid = float(pp_s[f"cext"].isel(time=0).mean(("draw", "chain")))
                color = to_hex(cmap(minmax_scaler(cid)) )
                hdi = az.hdi(pp_s, .95)

                for vi, data_variable in enumerate(data_variables):
                    y_mean = pp_s[data_variable].mean(dim=("chain", "draw"))
                    ax: plt.Axes = axes[vi, bi]

                    has_obs = not bool(ob_s[data_variable].isnull().all())
                    ls = "-" if has_obs else "dotted"
                    lw = 1.5 if has_obs else 1.0
                    alpha = .3 if has_obs else 0.15
                    alpha = alpha / len(sample_ids)
                    # c = "black" if bool(ob_s[data_variable].isnull().all()) else color
                    ax.fill_between(
                        post_pred[x_dim].values, *hdi[data_variable].values.T, 
                        alpha=alpha, color=color, zorder=2.001
                    )

                    ax.plot(
                        post_pred[x_dim].values, y_mean.values, lw=lw, 
                        color=color, ls=ls, zorder=2.003
                    )

                    ax.plot(
                        ob_s[x_dim].values, ob_s[data_variable].values, 
                        color=color, label=cid,
                        marker="o", ls="", ms=5, alpha=.75, zorder=2.003
                    )
                    # ax.legend()
                    if si == 0:
                        draw_axis_letter(ax, next(panelletter), loc=(0.9,0.97))
                        # ax.spines[["top", "right"]].set_visible(False)
                        ax.set_xlim(post_pred[x_dim].min()*0.95, post_pred[x_dim].max())
                        if data_variable == "nrf2":
                            ax.hlines(1, post_pred[x_dim].min()*0.95, post_pred[x_dim].max(), 
                                      color="black", lw=.25)
                        if data_variable == "nrf2" or data_variable == "survival":
                            iax = ax.inset_axes(bounds=(0.05, 0.05, 0.08, 0.2), transform=ax.transAxes)
                        else:
                            iax = ax.inset_axes(bounds=(0.05, 0.75, 0.08, 0.2), transform=ax.transAxes)
                        
                        cb = plt.colorbar(
                            mappable=cm.ScalarMappable(
                                norm=Normalize(vmin=cmin, vmax=cmax), 
                                cmap=cmap
                            ), 
                            cax=iax,
                            ticks=[np.ceil(cmin), np.floor(cmax)],
                        )
                        cb.ax.tick_params(labelsize=8)
                        cb.set_ticklabels([f"{int(t)} µmol/L" for t in cb.get_ticks()])
                    if bi == 0: ax.set_ylabel(pretty_labels[data_variable]) 
                    cext_unit = obs.attrs[f"cext_diuron"]
                    if vi == 0: ax.set_title(f"$C_{{ext}} \leq {bin}$ {cext_unit}", fontsize=10) 

                    if data_variable == "nrf2":
                        ax.set_ylim(0,limits["nrf2"][s])
                    if data_variable == "survival":
                        ax.set_ylim(-0.05, 1.05)
                    if data_variable == "cint":
                        ax.set_ylim(0, obs.cint.max()*1.25)

        fig.text(x=0.5, y=0.02, s="Time [hpf]")
        fig.subplots_adjust(left=0.08, right=0.99, bottom=0.08, top=0.95, wspace=0.1, hspace=0.1)

        fig.savefig(f"{sim.output_path}/combined_pps_figure_{s}.png")
        plt.close()

    sim.coordinates["time"] = old_time