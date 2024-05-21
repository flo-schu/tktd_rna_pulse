import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import rc
from posterior_analysis import (
    log, prepare_casestudy, postprocess_posterior
)
from pymob.utils.plot_helpers import plot_loghist
from toopy.plot import letterer, draw_axis_letter
from parameter_identifyability import format_parameter

rng = np.random.default_rng(1)
cm = 1/2.56  # convert cm to inch
rc("font", family='sans-serif', size=9)

# define scenarios
scenarios = {
    "rna_pulse_3_6c_substance_specific": {"title": "Substance specific", "sim": None}, 
    "rna_pulse_3_6c_substance_independent_rna_protein_module": {"title": "Substance independent", "sim": None}, 
}

# load data
for i, (scenario, sdict) in enumerate(scenarios.items()):
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

    idata = sim.inferer.idata
    parameters = list(idata.posterior.data_vars.keys())

    n = 20000
    n_per_chain = int(n / idata.posterior.sizes["chain"])
    random_draws = rng.integers(0, idata.posterior.sizes["draw"], n_per_chain)
    sdict["random_draws"] = random_draws

# plot func
def plot_single_parameter_pair(posterior, likelihood, par1, par2, axes=None):
    if axes is None:
        fig, axes = plt.subplots(2, 2, height_ratios=[1,3], width_ratios=[3,1])
        fig.subplots_adjust(hspace=0.0)
    

    assert axes.shape == (2,2)
    axes[0,1].set_axis_off()  # remove the top right corner axis

    theta_1_draws = posterior[par1].stack(sample=("chain","draw"))
    theta_2_draws = posterior[par2].stack(sample=("chain","draw"))
    loglik = likelihood.sum(("id", "time"))
    total_loglik = loglik.to_array().sum("variable")

    color = mpl.colormaps["plasma_r"]
    color = mpl.colormaps["gray_r"]

    ax_scatter = axes[1,0]
    ax_scatter.scatter(
        x=theta_1_draws, 
        y=theta_2_draws, 
        c=total_loglik, 
        s=5,
        alpha=.1,
        cmap=color,
    )
    ax_scatter.set_xscale("log")
    ax_scatter.set_yscale("log")
    ax_scatter.set_xlabel(f"{format_parameter(par1, textwrap='{}')}", fontsize=16, weight="bold")
    ax_scatter.set_ylabel(f"{format_parameter(par2, textwrap='{}')}", fontsize=16, weight="bold")
    xlim = ax_scatter.get_xlim()
    ylim = ax_scatter.get_ylim()

    ax_xhist = axes[0,0]
    plot_loghist(
        theta_1_draws, 
        bins=40,
        ax=ax_xhist, 
        color=color(255),
        decorate=False
    )
    ax_xhist.set_yticks([])
    ax_xhist.set_xticks([])
    ax_xhist.spines[["top", "left", "right"]].set_visible(False)
    ax_xhist.set_xlim(xlim)

    ax_yhist = axes[1,1]
    plot_loghist(
        theta_2_draws, 
        ax=ax_yhist, 
        bins=40,
        color=color(255),
        decorate=False,
        orientation="horizontal"
    )

    ax_yhist.set_xticks([])
    ax_yhist.set_yticks([])
    ax_yhist.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax_yhist.spines[["top", "right", "bottom"]].set_visible(False)
    ax_yhist.set_ylim(ylim)

    # turn of labels for minot ticks
    ax_scatter.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax_scatter.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    # set major labels to Sci Notation
    ax_scatter.xaxis.set_major_formatter(mpl.ticker.LogFormatterSciNotation())
    ax_scatter.yaxis.set_major_formatter(mpl.ticker.LogFormatterSciNotation())
    ax_scatter.tick_params(axis='both', which='major', labelsize=14)

# plot
sub = "diclofenac"
fig = plt.figure(figsize=(8, 4))
gs = fig.add_gridspec(2, 4)
for i, (scenario, sdict) in enumerate(scenarios.items()):
    idata = sdict["sim"].inferer.idata
    draws = sdict["random_draws"]
    posterior = idata.posterior.sel(draw=draws, substance=sub)
    likelihood = idata.log_likelihood.sel(draw=draws)


    gs_sub = gs[:, (i)*2:(i+1)*2].subgridspec(
        2, 2, height_ratios=[1,6], width_ratios=[6,1], 
        wspace=0, hspace=0
    )
    axes = gs_sub.subplots()
    par1 = "kk"
    par2 = "r_rt"
    plot_single_parameter_pair(
        posterior, likelihood, 
        par1=par1, par2=par2, 
        axes=axes
    )

    if par1 == "kk":
        axes[1,0].set_xlabel(f"$k_k$")
    if par2 == "kk":
        axes[1,0].set_ylabel(f"$k_k$")

    axes[0,0].set_title(sdict["title"], fontsize=16)

fig.subplots_adjust(wspace=0.6, bottom=0.15, left=0.12, right=0.97, top=0.92)
fig.savefig(f"results/plots/fig6_parameter_analysis_{sub}_{par1}_{par2}.png")

print("done!")





