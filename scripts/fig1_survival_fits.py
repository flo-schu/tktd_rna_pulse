import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import rc
from posterior_analysis import (
    log, prepare_casestudy, postprocess_posterior
)
from toopy.plot import letterer, draw_axis_letter

cm = 1/2.56  # convert cm to inch
rc("font", family='sans-serif', size=9)

# do not reload unless you want to recomp
# scenarios = {
#     "guts_scaled_damage": {"title": f"GUTS", "sim": None},
#     "guts_rna": {"title": "GUTS-RNA", "sim": None},
#     "rna_pulse_3_1_independent": {"title": "GUTS-RNA-pulse", "sim": None}, 
# }   

# scenarios = {
#     "guts_scaled_damage_fixed_sigma": {"title": f"GUTS", "sim": None},
#     "guts_rna_fixed_sigma": {"title": "GUTS-RNA", "sim": None},
#     "rna_pulse_3_6_substance_specific_fixed_sigma": {"title": "GUTS-RNA-pulse", "sim": None}, 
# }

scenarios = {
    "guts_scaled_damage": {"title": f"GUTS-scaled-damage", "sim": None},
    "guts_rna_2": {"title": "GUTS-RNA", "sim": None},
    "rna_pulse_3_6c_substance_specific": {"title": "GUTS-RNA-pulse", "sim": None}, 
}

substances = {
    "diuron": {
        "experiment_ids": [43, 44],
        "posterior_concs": [5.0, 14.0, 25.0, 60.0],
        "cbar_midpoint": 30
    }, 
    "diclofenac": {
        "experiment_ids": [40, 41],
        "posterior_concs": [1.0, 7.0, 12.0, 20.0, 50.0, 100.0],
        "cbar_midpoint": 13
    }, 
    "naproxen": {
        "experiment_ids": [47, 48, 49],
        "posterior_concs": [100, 260, 370, 500, 700, 1200],
        "cbar_midpoint": 650
    },
}
data_variable = "survival"
cmap = mpl.colormaps["cool"]

idatas = []
sims = []
fig, axes = plt.subplots(
    3, len(scenarios)+1, sharex=True, sharey=True,
    figsize=(20*cm, 14*cm),
    width_ratios=[1]*len(scenarios) + [0.25]
)
panelletter = letterer()

for i, (scenario, sdict) in enumerate(scenarios.items()):
    if sdict["sim"] is None:
        config = prepare_casestudy(
            case_study=("reversible_damage", scenario),
            config_file="settings.cfg",
            pkg_dir="case_studies"
        )
        from sim import SingleSubstanceSim2, xarray_indexer
        sim = SingleSubstanceSim2(config)

        sim.set_inferer("numpyro")
        sim.inferer.load_results("numpyro_posterior_filtered.nc")
        sdict["sim"] = sim
    else:
        sim = sdict["sim"]

    axes[0, i].set_title(sdict["title"], fontsize=14)

    # drop Diclofenac concentrations that make the colorscale meaningless 
    treatment_mask = ~sim.observations.treatment_id.isin([134, 136])
    sim.observations = sim.observations.where(treatment_mask, drop=True)
    sim.coordinates["time"] = np.linspace(24, 120, 100)
    for j, (substance, substance_dict) in enumerate(substances.items()):
        # exposed = sim.observations\
        #     .swap_dims(id="substance")\
        #     .sel(substance=substance)
        # ids_subset = exposed.experiment_id.values

        posterior_predictions = sim.inferer.posterior_predictions(
            n=2000, 
            seed=1
        )
        mask = sim.observations.experiment_id.isin(substance_dict["experiment_ids"])
        sdata = sim.observations.where(mask, drop=True)
        C = np.round(sdata.cext_nom.values, 1)
        norm = mpl.colors.TwoSlopeNorm(vmin=-5., vcenter=0., vmax=10)
        # norm = mpl.colors.Normalize(vmin=C.min(), vmax=C.max())
        norm = mpl.colors.TwoSlopeNorm(vmin=C.min(), vcenter=substance_dict["cbar_midpoint"], vmax=C.max())
        for eid in substance_dict["experiment_ids"]:
            indices = {
                "substance": substance, 
                "experiment_id": eid,
            }

            obs = xarray_indexer(
                ds=sim.observations, 
                indices=indices, 
                original_index="id"
            )
        
            if data_variable == "survival" or data_variable == "lethality":
                obs[data_variable] = obs[data_variable] / obs["nzfe"]
            else:
                obs = sim.observations

            for ii in obs.id.values:
                o = obs.sel(id=[ii])
                color = cmap(norm(float(o.cext_nom)))
                ax = sim.inferer.plot_predictions(
                    observations=o,
                    predictions=posterior_predictions.sel(id=[ii]),
                    data_variable=data_variable, 
                    x_dim="time",
                    ax=axes[j, i],
                    subset={},
                    mode="mean+hdi",
                    plot_options={
                        "obs": dict(ms=5, alpha=.7, color=color),
                        "pred_mean": dict(color=None, ls=""),
                        "pred_hdi": dict(color=None, ls="", alpha=0),
                    }
                )

        # plot predictions nearest to concentrations defined
        # above
        for conc in substance_dict["posterior_concs"]:
            preds = sdata.swap_dims(id="cext_nom")\
                .sortby("cext_nom")\
                .sel(cext_nom=[conc], method="nearest")\
                .swap_dims(cext_nom="id")

            obs = sim.observations.sel(id=preds.id)
            if data_variable == "survival" or data_variable == "lethality":
                obs[data_variable] = obs[data_variable] / obs["nzfe"]
            else:
                obs = sim.observations

            for ii in preds.id.values:
                p = preds.sel(id=[ii])
                color = cmap(norm(float(p.cext_nom)))
                ax = sim.inferer.plot_predictions(
                    observations=obs.sel(id=[ii]), # provide pseudo obs
                    predictions=posterior_predictions.sel(id=[ii]),
                    data_variable=data_variable, 
                    x_dim="time",
                    ax=axes[j, i],
                    subset={},
                    mode="mean+hdi",
                    plot_options={
                        "obs": dict(ms=0, alpha=.7, color=None),
                        "pred_mean": dict(color=color),
                        "pred_hdi": dict(color=color),
                    }
                )
        
        draw_axis_letter(axes[j,i], next(panelletter), loc=(0.02,0.9))
        axes[j, 0].set_ylabel(f"{data_variable.capitalize()}")
        if i > 0:
            axes[j, i].set_ylabel("")

        if i == len(scenarios) - 1:
            axes[j, -1].axis("off")
            cbar = fig.colorbar(
                mappable=mpl.cm.ScalarMappable(norm=norm, cmap=cmap), 
                ax=axes[j, -1],
                fraction=1.0,
                pad=0,
                aspect=10,
                label=f"{substance.capitalize()} [µmol/L]"
            )

            ax.text(-0.42, 0.5, substance.capitalize(), 
                    transform=axes[j, 0].transAxes,
                    rotation=90, va="center", fontsize=14)
    
    axes[0, i].set_xlabel("")
    axes[1, i].set_xlabel("")
    axes[2, i].set_xlabel("Time [h]")

fig.subplots_adjust(bottom=0.1, top=0.9, right=0.95, left=0.12, wspace=0.1, hspace=0.1)
fig.savefig(f"results/plots/fig1_{data_variable}_model_comparison_{len(scenarios)}_models.png", dpi=150)