import os
import numpy as np
from matplotlib import pyplot as plt
import xarray as xr
import click
from click.testing import CliRunner
from functools import partial
from frozendict import frozendict
import jax
import jax.numpy as jnp

from pymob import Config

def likelihood_landscapes(sim, parx, pary, std_dev, n_grid_points, n_vector_points, sparse, idata_file, center_at_posterior_mode):

    folder = os.path.join(sim.output_path, "likelihood_landscapes")
    os.makedirs(folder, exist_ok=True)

    sim.config.inference_numpyro.gaussian_base_distribution = True
    sim.config.jaxsolver.throw_exception = False
    sim.config.jaxsolver.max_steps = 10_000

    # set up the bounds for likelihood landscapes
    # this must be done here
    sim.config.model_parameters[parx].min = None
    sim.config.model_parameters[parx].max = None
    sim.config.model_parameters[pary].min = None
    sim.config.model_parameters[pary].max = None

    sim.dispatch_constructor()
    sim.set_inferer("numpyro")
    sim.inferer.load_results(idata_file)
    
    if hasattr(sim.inferer.idata, "posterior"):
        group = "posterior"
    elif hasattr(sim.inferer.idata, "prior"):
        group = "prior"
    else:
        raise AttributeError("idata has neither prior nor posterior.")

    def dataset_to_dict(dataset: xr.Dataset):
        return {k: v.values for k, v in dataset.data_vars.items()}
    
    mode_draw = sim.inferer.idata.log_likelihood.sum(("id", "time")).to_array().sum("variable").argmax()
    
    mode = dataset_to_dict(sim.inferer.idata[f"unconstrained_{group}"].sel(chain=0, draw=mode_draw))
    mpx = np.array(mode[f"{parx}_normal_base"], ndmin=1)
    mpy = np.array(mode[f"{pary}_normal_base"], ndmin=1)
    mode_frozen = frozendict({k: tuple(np.array(v, ndmin=1).tolist()) for k, v in mode.items()})

    # get extra dim
    parx_extra_dim = [d for d in sim.inferer.idata[group][parx].dims if d not in ("chain", "draw")]
    if len(parx_extra_dim) == 0:
        parx_coords = [""]
        parx_extra_dim = ""
    elif len(parx_extra_dim) == 1:
        parx_coords = [f" [{v}]" for v in sim.inferer.idata[group][parx].coords[parx_extra_dim[0]].values]
        parx_extra_dim = parx_extra_dim[0]
    else:
        raise ValueError("Only 1-D Parameters are supported")

    # get extra dim
    pary_extra_dim = [d for d in sim.inferer.idata[group][pary].dims if d not in ("chain", "draw")]
    if len(pary_extra_dim) == 0:
        pary_coords = [""]
        pary_extra_dim = ""
    elif len(pary_extra_dim) == 1:
        pary_coords = [f" [{v}]" for v in sim.inferer.idata[group][pary].coords[pary_extra_dim[0]].values]
        pary_extra_dim = pary_extra_dim[0]
    else:
        raise ValueError("Only 1-D Parameters are supported")

    if sparse:
        nrows = 1
    else:
        nrows = len(mpy)

    fig, axes = plt.subplots(ncols=len(mpx), nrows=nrows, figsize=(2+len(mpx)*4,2+nrows*3), squeeze=False)

    for i, (axcol, x) in enumerate(zip(axes.T, mpx)):
        for j, y in enumerate(mpy):
            if sparse:
                if i != j:
                    # skip the execution of combinations that are not on the diagonal
                    continue
                else:
                    ax = axcol[0]
            else:
                ax = axcol[j]

            sim.dispatch_constructor()
            sim.set_inferer("numpyro")

            def only_data_loglik(joint, prior, data): 
                return jnp.array(list({k: v.sum() for k,v in data.items()}.values())).sum()

            f, grad = sim.inferer.create_log_likelihood(
                return_type="custom",
                custom_return_fn=only_data_loglik,
                check=False,
                scaled=True,
                vectorize=False,
                # gradients=True
            )

            @partial(jax.jit, static_argnames=["_i","_j", "_mode", "_parx", "_pary"])
            def func(theta, _i, _j, _mode, _parx, _pary):
                # select first substance
                params = {}
                for key, _value in _mode.items():
                    value = jnp.array(_value)
                    if key in theta and theta[key].shape == value.shape:
                        new_value = theta[key]
                        params.update({key: jnp.array(new_value)})
                    elif key == f"{_parx}_normal_base":
                        value = value.at[_i].set(theta[key][0])
                        params.update({key: value})
                    elif key == f"{_pary}_normal_base":
                        value = value.at[_j].set(theta[key][0])
                        params.update({key: value})
                    else:
                        params.update({key: jnp.array(value)})
                    
                return f(params)
            
            # Compute the gradient function
            grad = jax.grad(partial(func, _i=i, _j=j, _mode=mode_frozen, _parx=parx, _pary=pary))
            func_ = partial(func, _i=i, _j=j, _mode=mode_frozen, _parx=parx, _pary=pary)

            if center_at_posterior_mode:
                bounds=([x - std_dev, x + std_dev], [y - std_dev, y + std_dev])
            else:
                bounds=([0 - std_dev, 0 + std_dev], [0 - std_dev, 0 + std_dev])
            
            ax = sim.inferer.plot_likelihood_landscape(
                parameters=(parx, pary),
                bounds=bounds,
                log_likelihood_func=func_,
                gradient_func=None if n_vector_points== 0 else grad,
                n_grid_points=n_grid_points,
                n_vector_points=n_vector_points,
                normal_base=True,
                ax=ax
            )

            ax.set_xlabel(f"{parx.replace(f'_{parx_extra_dim}', '')}{parx_coords[i]}")
            ax.set_ylabel(f"{pary.replace(f'_{pary_extra_dim}', '')}{pary_coords[j]}")

            fig.tight_layout()
            ax.figure.savefig(os.path.join(folder, f"{parx}__{pary}.png"))

@click.command()
@click.option("--config", type=str)
@click.option("--parx", type=str)
@click.option("--pary", type=str)
@click.option("--std_dev", type=float, default=2)
@click.option("--n_grid_points", type=int, default=50)
@click.option("--n_vector_points", type=int, default=0)
@click.option("--debug/--no-debug", default=False)
@click.option("--sparse/--full", default=False)
@click.option("--idata_file", default=None)
@click.option("--center-at-posterior-mode/--center-at-prior-mode", default=False)
def main(config, parx, pary, std_dev, n_grid_points, n_vector_points, debug, sparse, idata_file, center_at_posterior_mode):

    if debug:
        import pdb
        pdb.set_trace()

    cfg = Config(config)
    cfg.import_casestudy_modules()
    Simulation = cfg.import_simulation_from_case_study()
    sim = Simulation(config)
    sim.setup()

    likelihood_landscapes(
        sim=sim,
        parx=parx,
        pary=pary,
        std_dev=std_dev,
        n_grid_points=n_grid_points,
        n_vector_points=n_vector_points,
        sparse=sparse,
        idata_file=idata_file,
        center_at_posterior_mode=center_at_posterior_mode
    )



if __name__ == "__main__":
    if bool(os.getenv("debug")):
        runner = CliRunner(echo_stdin=True)
        result = runner.invoke(main, catch_exceptions=False, args=[
            "--config=scenarios/rna_pulse_4_substance_specific/settings.cfg",
            # using --no-debug is important here, because otherwise the pdb interferes
            # with the vscode call I suspect.
            "--parx=k_i_substance",
            "--pary=r_rt_substance",
            "--n_grid_points=5",
            "--n_vector_points=5",
            "--no-debug",
            "--full"
        ])
        if isinstance(result.exception, SystemExit):
            raise KeyError(
                "Invokation of the click command did not execute correctly. " +
                f"Recorded output: {' '.join(result.output.splitlines())}"
            )
        
        else:
            print(result.output)
            
        
    else:
        main()
