from functools import partial

import jax.numpy as jnp
import jax
from diffrax import (
    diffeqsolve, 
    Dopri5, 
    ODETerm, 
    SaveAt, 
    PIDController, 
    RecursiveCheckpointAdjoint
)

from pymob.sim.solvetools import mappar

def simplified_ode_solver(model, post_processing, parameters, coordinates, indices, data_variables, n_ode_states, seed=None):
    coords = coordinates
    y0_arr = parameters["y0"]
    params = parameters["parameters"]
    time = tuple(coords["time"])

    # collect parameters
    ode_args = mappar(model, params, exclude=["t", "X"])
    pp_args = mappar(post_processing, params, exclude=["t", "time", "results"])

    # index parameter according to substance
    s_idx = indices["substance"].values
    ode_args = [jnp.array(a, ndmin=1)[s_idx] for a in ode_args]
    pp_args = [jnp.array(a, ndmin=1)[s_idx] for a in pp_args]

    # transform Y0 to jnp.arrays
    Y0 = [jnp.array(v.values) for _, v in y0_arr.data_vars.items()]

    initialized_eval_func = partial(
        odesolve_splitargs,
        model=model,
        post_processing=post_processing,
        time=time,
        odestates = tuple(y0_arr.keys()),
        n_odeargs=len(ode_args),
        n_ppargs=len(pp_args),

    )
        
    loop_eval = jax.vmap(
        initialized_eval_func, 
        in_axes=(
            *[0 for _ in range(n_ode_states)], 
            *[0 for _ in range(len(ode_args))],
            *[0 for _ in range(len(pp_args))],
        )
    )
    result = loop_eval(*Y0, *ode_args, *pp_args)
    return result


@partial(jax.jit, static_argnames=["model", "post_processing", "time", "odestates", "n_odeargs", "n_ppargs"])
def odesolve_splitargs(*args, model, post_processing, time, odestates, n_odeargs, n_ppargs):
    n_odestates = len(odestates)
    y0 = args[:n_odestates]
    odeargs = args[n_odestates:n_odeargs+n_odestates]
    ppargs = args[n_odeargs+n_odestates:n_odeargs+n_odestates+n_ppargs]
    sol = odesolve(model=model, y0=y0, time=time, args=odeargs)
    
    res_dict = {v:val for v, val in zip(odestates, sol)}

    return post_processing(res_dict, jnp.array(time), *ppargs)


@partial(jax.jit, static_argnames=["model"])
def odesolve(model, y0, time, args):
    f = lambda t, y, args: model(t, y, *args)
    
    term = ODETerm(f)
    solver = Dopri5()
    saveat = SaveAt(ts=time)
    stepsize_controller = PIDController(rtol=1e-5, atol=1e-7)
    t_min = jnp.array(time).min()
    t_max = jnp.array(time).max()

    sol = diffeqsolve(
        terms=term, 
        solver=solver, 
        t0=t_min, 
        t1=t_max, 
        dt0=0.1, 
        y0=tuple(y0), 
        args=args, 
        saveat=saveat, 
        stepsize_controller=stepsize_controller,
        adjoint=RecursiveCheckpointAdjoint(),
        max_steps=10**5,
        # throw=False returns inf for all t > t_b, where t_b is the time 
        # at which the solver broke due to reaching max_steps. This behavior
        # happens instead of throwing an exception.
        throw=False,   
    )
    
    return list(sol.ys)


def calculate_psurv2(results, t, interpolation, z, kk, h_b):
    # calculate survival 
    p_surv = survival_jax(t, results["nrf2"], z, kk, h_b)
    results["survival"] = p_surv
    results["lethality"] = 1 - p_surv
    return results


@jax.jit
def survival_jax(t, damage, z, kk, h_b):
    """
    survival probability derived from hazard 
    first calculate cumulative Hazard by integrating hazard cumulatively over t
    then calculate the resulting survival probability
    It was checked that `survival_jax` behaves exactly the same as `survival`
    """
    hazard = kk * jnp.where(damage - z < 0, 0, damage - z) + h_b
    # H = jnp.array([jax.scipy.integrate.trapezoid(hazard[:i+1], t[:i+1]) for i in range(len(t))])
    H = jnp.array([jnp.trapz(hazard[:i+1], t[:i+1], axis=0) for i in range(len(t))])
    S = jnp.exp(-H)

    return S


def tktd_rna_3_6c(t, X, r_0, k_i, r_rt, r_rd, z_ci, v_rt, k_p, k_m, ci_max):
    """
    A simplified RNA pulse model.

    This function models gene expression and metabolization of the internal
    concentration of a substance. The gene expression is controlled by a arctan
    step function based on the internal concentration (Ci) and switches on the
    gene's expression. The gene then translates a Protein, which metabolizes 
    the internal concentration proportional to its expression level. The concept
    of protein must be understood not as a single Protein but as a collection
    of detoxification measures, which reduce the internal concentration of the
    compound and keep it at a reasonable level.

    Changes w.r.t. to RNA 3.6 model
    -------------------------------
    - The model removes the b_base parameter and replaces it with a fixed R_0
      parameter

    Parameters
    ----------
    t : float
        Timestep at which the model is evaluated.

    X : tuple
        A tuple containing three elements: 
        - Ce : float
            The external concentration.
        - Ci : float
            The internal concentration.
        - R : float
            The gene expression level.
        - P : float
            The protein level.

    r_0 : float
        Initial value of the gene expression level.        
    
    k_i : float
        Internal consumption rate constant.

    k_m : float
        Metabolization rate constant.
    
    r_rt : float
        Maximum gene expression rate constant. Termed k_rt in the paper

    r_rd : float
        Gene degradation rate constant. Termed k_rd in the paper

    z_ci : float
        The threshold for gene expression.

    v_rt : float, optional
        The slope parameter for the inverse tangent step function. This
        parameter regulates the responsiveness of the gene expression induction.

    k_p : float
        Dominant protein translation rate konstant.

    Returns
    -------
    dCe_dt : float
        The rate of change of external concentration.

    dCi_dt : float
        The rate of change of internal concentration.

    dR_dt : float
        The rate of change of gene expression level.

    dR_dt : float
        The rate of change of protein level.

    """
    Ce, Ci, R, P = X

    active = 0.5 + (1 / jnp.pi) * jnp.arctan(v_rt * (Ci / ci_max - z_ci))

    dCe_dt = 0.0
    dCi_dt = Ce * k_i - Ci * P * k_m
    dR_dt = r_rt * active - (R - r_0) * r_rd
    dP_dt = k_p * ((R - r_0) - P)

    return dCe_dt, dCi_dt, dR_dt, dP_dt

def tktd_rna_4(t, X, r_0, k_i, r_rt, r_rd, z_ci, v_rt, k_p, k_m, h_b, kk, z, ci_max):
    """
    A simplified RNA pulse model.

    This function models gene expression and metabolization of the internal
    concentration of a substance. The gene expression is controlled by a arctan
    step function based on the internal concentration (Ci) and switches on the
    gene's expression. The gene then translates a Protein, which metabolizes 
    the internal concentration proportional to its expression level. The concept
    of protein must be understood not as a single Protein but as a collection
    of detoxification measures, which reduce the internal concentration of the
    compound and keep it at a reasonable level.

    Changes w.r.t. to RNA 3.6 model
    -------------------------------
    - The model uses a sigmoid switch function instead of arctan
    - The model computes the hazard from the threshold model directly instead
      of the post processing

    Parameters
    ----------
    t : float
        Timestep at which the model is evaluated.

    X : tuple
        A tuple containing three elements: 
        - Ce : float
            The external concentration.
        - Ci : float
            The internal concentration.
        - R : float
            The gene expression level.
        - P : float
            The protein level.
        - H : float
            The cumulative hazard.
        - P : float
            The survival probability.

    r_0 : float
        Initial value of the gene expression level.        
    
    k_i : float
        Internal consumption rate constant.

    k_m : float
        Metabolization rate constant.
    
    r_rt : float
        Maximum gene expression rate constant. Termed k_rt in the paper

    r_rd : float
        Gene degradation rate constant. Termed k_rd in the paper

    z_ci : float
        The threshold for gene expression.

    v_rt : float, optional
        The slope parameter for the inverse tangent step function. This
        parameter regulates the responsiveness of the gene expression induction.

    k_p : float
        Dominant protein translation rate konstant.

    Returns
    -------
    dCe_dt : float
        The rate of change of external concentration.

    dCi_dt : float
        The rate of change of internal concentration.

    dR_dt : float
        The rate of change of gene expression level.

    dR_dt : float
        The rate of change of protein level.

    dH_dt : float
        The hazard rate h(t) = b * max(D, 0) + h_b
    
    dS_dt : float
        The rate of change of the survival probability 
    """
    Ce, Ci, R, P, H, S = X

    # active = 0.5 + (1 / jnp.pi) * jnp.arctan(v_rt * (Ci / ci_max - z_ci))
    active = 1 / (1 + jnp.exp(- v_rt * (Ci/ci_max - z_ci)))

    dCe_dt = 0.0
    dCi_dt = Ce * k_i - Ci * P * k_m
    dR_dt = r_rt * active - (R - r_0) * r_rd
    dP_dt = k_p * ((R - r_0) - P)

    dH_dt = kk * jnp.maximum(R - z, jnp.array([0.0], dtype=float)) + h_b
    dS_dt = -dH_dt * S

    return dCe_dt, dCi_dt, dR_dt, dP_dt, dH_dt, dS_dt


def tktd_rna_5(t, X, r_0, k_i, r_rt, r_rd, z_ci, v_rt, k_p, k_m, h_b, kk, z, ci_max):
    """
    A simplified RNA pulse model.

    This function models gene expression and metabolization of the internal
    concentration of a substance. The gene expression is controlled by a arctan
    step function based on the internal concentration (Ci) and switches on the
    gene's expression. The gene then translates a Protein, which metabolizes 
    the internal concentration proportional to its expression level. The concept
    of protein must be understood not as a single Protein but as a collection
    of detoxification measures, which reduce the internal concentration of the
    compound and keep it at a reasonable level.

    Changes w.r.t. to RNA 4 model
    -------------------------------
    - The model does not evolve the survival probability S over time any more
      This is done more efficiently in the post processing, by simply taking the 
      exponent


    Parameters
    ----------
    t : float
        Timestep at which the model is evaluated.

    X : tuple
        A tuple containing three elements: 
        - Ce : float
            The external concentration.
        - Ci : float
            The internal concentration.
        - R : float
            The gene expression level.
        - P : float
            The protein level.
        - H : float
            The cumulative hazard.

    r_0 : float
        Initial value of the gene expression level.        
    
    k_i : float
        Internal consumption rate constant.

    k_m : float
        Metabolization rate constant.
    
    r_rt : float
        Maximum gene expression rate constant. Termed k_rt in the paper

    r_rd : float
        Gene degradation rate constant. Termed k_rd in the paper

    z_ci : float
        The threshold for gene expression.

    v_rt : float, optional
        The slope parameter for the inverse tangent step function. This
        parameter regulates the responsiveness of the gene expression induction.

    k_p : float
        Dominant protein translation rate konstant.

    Returns
    -------
    dCe_dt : float
        The rate of change of external concentration.

    dCi_dt : float
        The rate of change of internal concentration.

    dR_dt : float
        The rate of change of gene expression level.

    dR_dt : float
        The rate of change of protein level.
    
    dH_dt : float
        The hazard rate h(t) = b * max(D, 0) + h_b
    """
    Ce, Ci, R, P, H = X

    # active = 0.5 + (1 / jnp.pi) * jnp.arctan(v_rt * (Ci / ci_max - z_ci))
    active = 1 / (1 + jnp.exp(- v_rt * (Ci/ci_max - z_ci)))

    dCe_dt = 0.0
    dCi_dt = Ce * k_i - Ci * P * k_m
    dR_dt = r_rt * active - (R - r_0) * r_rd
    dP_dt = k_p * ((R - r_0) - P)

    dH_dt = kk * jnp.maximum(R - z, jnp.array([0.0], dtype=float)) + h_b

    return dCe_dt, dCi_dt, dR_dt, dP_dt, dH_dt

def survival(results, t, interpolation):
    results["survival"] = jnp.exp(-results["H"])
    return results



def tktd_rna_6(t, X, r_0, k_i, k_e, r_rt, r_rd, z_ci, v_rt, k_p, k_m, h_b, kk, z, ci_max):
    """
    A simplified RNA pulse model.

    This function models gene expression and metabolization of the internal
    concentration of a substance. The gene expression is controlled by a arctan
    step function based on the internal concentration (Ci) and switches on the
    gene's expression. The gene then translates a Protein, which metabolizes 
    the internal concentration proportional to its expression level. The concept
    of protein must be understood not as a single Protein but as a collection
    of detoxification measures, which reduce the internal concentration of the
    compound and keep it at a reasonable level.

    Changes w.r.t. to RNA 5 model
    -------------------------------
    - The model includes a term for passive degradation

    Parameters
    ----------
    t : float
        Timestep at which the model is evaluated.

    X : tuple
        A tuple containing three elements: 
        - Ce : float
            The external concentration.
        - Ci : float
            The internal concentration.
        - R : float
            The gene expression level.
        - P : float
            The protein level.

    r_0 : float
        Initial value of the gene expression level.        
    
    k_i : float
        Uptake rate constant.

    k_e : float
        Passive elimination rate constant.

    k_m : float
        Metabolization rate constant.
    
    r_rt : float
        Maximum gene expression rate constant. Termed k_rt in the paper

    r_rd : float
        Gene degradation rate constant. Termed k_rd in the paper

    z_ci : float
        The threshold for gene expression.

    v_rt : float, optional
        The slope parameter for the inverse tangent step function. This
        parameter regulates the responsiveness of the gene expression induction.

    k_p : float
        Dominant protein translation rate konstant.

    Returns
    -------
    dCe_dt : float
        The rate of change of external concentration.

    dCi_dt : float
        The rate of change of internal concentration.

    dR_dt : float
        The rate of change of gene expression level.

    dR_dt : float
        The rate of change of protein level.

    """
    Ce, Ci, R, P, H = X

    # active = 0.5 + (1 / jnp.pi) * jnp.arctan(v_rt * (Ci / ci_max - z_ci))
    active = 1 / (1 + jnp.exp(- v_rt * (Ci/ci_max - z_ci)))

    dCe_dt = 0.0
    dCi_dt = k_i * Ce  - Ci * P * k_m - k_e * Ci
    dR_dt = r_rt * active - (R - r_0) * r_rd
    dP_dt = k_p * ((R - r_0) - P)

    dH_dt = kk * jnp.maximum(R - z, jnp.array([0.0], dtype=float)) + h_b

    return dCe_dt, dCi_dt, dR_dt, dP_dt, dH_dt