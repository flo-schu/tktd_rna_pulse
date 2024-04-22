import numpyro
from numpyro import distributions as dist
from pymob.inference.numpyro_backend import LogNormalTrans
import jax
import jax.numpy as jnp


def preprocessing(obs, masks):

    # indexes all observations that are not NAN
    obs_idx = {k:jnp.where(~jnp.isnan(v)) for k, v in obs.items()}

    # substance index of all non-NAN observations
    si_cext = jnp.broadcast_to(jnp.array([0, 1, 2]), obs["cext"].shape)[obs_idx["cext"]]
    si_cint = jnp.broadcast_to(jnp.array([0, 1, 2]), obs["cint"].shape)[obs_idx["cint"]]

    return {
        "obs": obs,
        "masks": masks,
        "obs_idx": obs_idx,
        "si_cext": si_cext,
        "si_cint": si_cint,
    }

def indexer(sim, obs, data_var, idx):
    sim_indexed = sim[data_var][*idx[data_var]]
    obs_indexed = obs[data_var][*idx[data_var]]
    return sim_indexed, obs_indexed


def lognormal_prior(name, loc, scale, normal_base=False):
    loc = jnp.array(loc)
    scale = jnp.array(scale)
    if normal_base:
        prior_norm = numpyro.sample(
            name=f"{name}_norm",
            fn=dist.Normal(loc=jnp.zeros_like(loc), scale=jnp.ones_like(scale))
        )

        prior = numpyro.deterministic(
            name=name,
            value=jnp.exp(prior_norm * scale + jnp.log(loc))
        )
    
    else:
        prior = numpyro.sample(
            name=name,
            fn=dist.LogNormal(loc=jnp.log(loc), scale=scale)
        )

    return prior

def halfnormal_prior(name, loc, scale, normal_base=False):
    loc = jnp.array(loc)
    scale = jnp.array(scale)
    if normal_base:
        prior_norm = numpyro.sample(
            name=f"{name}_norm",
            fn=dist.Normal(loc=jnp.zeros_like(loc), scale=jnp.ones_like(scale))
        )

        prior = numpyro.deterministic(
            name=name,
            value=loc + scale * jnp.abs(prior_norm)
        )
    
    else:
        prior = numpyro.sample(
            name=name,
            fn=dist.HalfNormal(scale=scale)
        )

    return prior


# RNA pulse model 3.6c 
# ====================
# removes b_base parameter

def model_rna_pulse_3_6c_substance_independent_rna_protein_module(solver, obs, masks, ci_max, only_prior=False):
    """Probability model with substance specific parameters and a conditional 
    binomial probability model for survival.
    
    Description of the probability model
    ------------------------------------
    The model is written so that MCMC samples are drawn from standard normal
    distributions, which are then deterministically mapped onto the log-normal
    space. This is helpful for using stochastic variational inference (SVI) with
    a multivariate normal target distribution. This target distribution can then
    in theory be used as a sampling distribution to generate proposals for true
    MCMC with NUTS.

    The model than inputs the parameters into the ODE solver, which maps the 
    parameters onto the different experiments with the different substances. And
    calculates ODE solutions for all experiments entered into the Simulation.
    Finally the log-probabilities of the observations (masked to exclude missing
    observations) are compared given the solutions of the ODE and the sigmas
    of the error model.

    Model parameters
    ----------------
    k_i: uptake rate
    r_rt: gene expression rate
    r_rd: RNA decay rate
    v_rt: Speed of on-switch of gene expression
    z_ci: internal concentration threshold for NRF2 activation 
    r_pt: Metabolization protein translation rate from RNA expression
    r_pd: Protein decay rate 
    h_b: background hazard
    z: hazard threshold according to GUTS model
    kk: killing rate of the survival model
    sigma_*: scale parameter of the error-distributions.

    Dependency structure
    --------------------
    The following parameters are assumed as independent w.r.t to the substances:
    - r_rt
    - r_rd
    - k_p
    - z
    - v_rt
    - kk
    - h_b
    - sigma_cint
    - sigma_nrf2
    """
    EPS = 1e-10

    # substance specific parameters
    k_i    = lognormal_prior(name="k_i"   , scale=2, loc=[1.0  , 1.0 , 1.0   ], normal_base=True)
    k_m    = lognormal_prior(name="k_m"   , scale=2, loc=[0.05 , 0.05, 0.05  ], normal_base=True)
    z_ci   = lognormal_prior(name="z_ci"  , scale=2, loc=[0.5  , 0.5 , 0.5   ], normal_base=True)
        
    # general parameters
    r_rt   = lognormal_prior(name="r_rt"  , scale=2, loc=[1.0  ], normal_base=True)
    r_rd   = lognormal_prior(name="r_rd"  , scale=2, loc=[0.5  ], normal_base=True)
    k_p    = lognormal_prior(name="k_p"   , scale=2, loc=[0.02 ], normal_base=True)
    z      = lognormal_prior(name="z"     , scale=2, loc=[1    ], normal_base=True)
    v_rt   = lognormal_prior(name="v_rt"  , scale=2, loc=[10   ], normal_base=True)
    h_b    = lognormal_prior(name="h_b"   , scale=2, loc=[1e-8 ], normal_base=True)
    kk     = lognormal_prior(name="kk"    , scale=2, loc=[0.02 ], normal_base=True)

    # error model sigmas
    sigma_cint = halfnormal_prior(name="sigma_cint", loc=0, scale=[5.0 ], normal_base=True)
    sigma_nrf2 = halfnormal_prior(name="sigma_nrf2", loc=0, scale=[5.0 ], normal_base=True)

    ci_max_ = numpyro.deterministic("ci_max", ci_max)

    if only_prior:
        # stop early if only the prior should be sampled
        return

    # deterministic model
    theta = {
        "k_i": k_i,
        "k_m": k_m,
        "r_rt": r_rt,
        "r_rd": r_rd,
        "v_rt": v_rt,
        "z_ci": z_ci,
        # Ci_max must have the same order as ot
        "ci_max": ci_max_,
        "k_p": k_p,
        "h_b": h_b,
        "z": z,
        "kk": kk,
    }

    sim = solver(theta=theta)
    cext = numpyro.deterministic("cext", sim["cext"])
    cint = numpyro.deterministic("cint", sim["cint"])
    P = numpyro.deterministic("P", sim["P"])
    nrf2 = numpyro.deterministic("nrf2", sim["nrf2"])
    leth = numpyro.deterministic("lethality", sim["lethality"])
    surv = numpyro.deterministic("survival", sim["survival"])

    # indexing
    substance_idx = obs["substance_index"]
    sigma_cint_indexed = sigma_cint[substance_idx]
    sigma_nrf2_indexed = sigma_nrf2[substance_idx]

    sigma_cint_ix_bc = jnp.broadcast_to(sigma_cint_indexed.reshape((-1, 1)), obs["cint"].shape)
    sigma_nrf2_ix_bc = jnp.broadcast_to(sigma_nrf2_indexed.reshape((-1, 1)), obs["nrf2"].shape)

    # error model
    S = jnp.clip(surv, EPS, 1 - EPS) 
    S_cond = S[:, 1:] / S[:, :-1]
    S_cond_ = jnp.column_stack([jnp.ones_like(substance_idx), S_cond])

    n_surv = obs["survivors_before_t"]
    S_mask = masks["survival"]
    obs_survival = obs["survival"]
    
    numpyro.sample("cint_obs", dist.LogNormal(loc=jnp.log(cint + EPS), scale=sigma_cint_ix_bc).mask(masks["cint"]), obs=obs["cint"])
    numpyro.sample("nrf2_obs", dist.LogNormal(loc=jnp.log(nrf2), scale=sigma_nrf2_ix_bc).mask(masks["nrf2"]), obs=obs["nrf2"])    
    numpyro.sample(
        "survival_obs", 
        dist.Binomial(probs=S_cond_, total_count=n_surv).mask(S_mask), 
        obs=obs_survival
    )



def model_rna_pulse_3_6c_substance_specific(solver, obs, masks, ci_max, only_prior=False):
    """Probability model with substance specific parameters and a conditional 
    binomial probability model for survival.
    
    Description of the probability model
    ------------------------------------
    The model is written so that MCMC samples are drawn from standard normal
    distributions, which are then deterministically mapped onto the log-normal
    space. This is helpful for using stochastic variational inference (SVI) with
    a multivariate normal target distribution. This target distribution can then
    in theory be used as a sampling distribution to generate proposals for true
    MCMC with NUTS.

    The model than inputs the parameters into the ODE solver, which maps the 
    parameters onto the different experiments with the different substances. And
    calculates ODE solutions for all experiments entered into the Simulation.
    Finally the log-probabilities of the observations (masked to exclude missing
    observations) are compared given the solutions of the ODE and the sigmas
    of the error model.

    Model parameters
    ----------------
    k_i: uptake rate
    r_rt: gene expression rate
    r_rd: RNA decay rate
    v_rt: Speed of on-switch of gene expression
    z_ci: internal concentration threshold for NRF2 activation 
    r_pt: Metabolization protein translation rate from RNA expression
    r_pd: Protein decay rate 
    h_b: background hazard
    z: hazard threshold according to GUTS model
    kk: killing rate of the survival model
    sigma_*: scale parameter of the error-distributions.

    Dependency structure
    --------------------
    The following parameters are assumed the same for all substances:
    """
    EPS = 1e-10

    # substance specific parameters
    k_i    = lognormal_prior(name="k_i"   , scale=2, loc=[1.0  , 1.0 , 1.0   ], normal_base=True)
    r_rt   = lognormal_prior(name="r_rt"  , scale=2, loc=[1.0  , 1.0 , 1.0   ], normal_base=True)
    r_rd   = lognormal_prior(name="r_rd"  , scale=2, loc=[0.5  , 0.5 , 0.5   ], normal_base=True)
    z_ci   = lognormal_prior(name="z_ci"  , scale=2, loc=[0.5  , 0.5 , 0.5   ], normal_base=True)
    k_p    = lognormal_prior(name="k_p"   , scale=2, loc=[0.02 , 0.02, 0.02  ], normal_base=True)
    k_m    = lognormal_prior(name="k_m"   , scale=2, loc=[0.05 , 0.05, 0.05  ], normal_base=True)
    z      = lognormal_prior(name="z"     , scale=2, loc=[1    , 1   , 1     ], normal_base=True)
    
    # general parameters
    v_rt   = lognormal_prior(name="v_rt"  , scale=2, loc=[10  , 10  , 10  ], normal_base=True)
    h_b    = lognormal_prior(name="h_b"   , scale=2, loc=[1e-8, 1e-8, 1e-8], normal_base=True)
    kk     = lognormal_prior(name="kk"    , scale=2, loc=[0.02, 0.02, 0.02], normal_base=True)

    # error model sigmas
    sigma_cint = halfnormal_prior(name="sigma_cint", loc=0, scale=[5, 5, 5], normal_base=True)
    sigma_nrf2 = halfnormal_prior(name="sigma_nrf2", loc=0, scale=[5, 5, 5], normal_base=True)

    ci_max_ = numpyro.deterministic("ci_max", ci_max)

    if only_prior:
        # stop early if only the prior should be sampled
        return

    # deterministic model
    theta = {
        "k_i": k_i,
        "k_m": k_m,
        "r_rt": r_rt,
        "r_rd": r_rd,
        "v_rt": v_rt,
        "z_ci": z_ci,
        # Ci_max must have the same order as ot
        "ci_max": ci_max_,
        "k_p": k_p,
        "h_b": h_b,
        "z": z,
        "kk": kk,
    }

    sim = solver(theta=theta)
    cext = numpyro.deterministic("cext", sim["cext"])
    cint = numpyro.deterministic("cint", sim["cint"])
    P = numpyro.deterministic("P", sim["P"])
    nrf2 = numpyro.deterministic("nrf2", sim["nrf2"])
    leth = numpyro.deterministic("lethality", sim["lethality"])
    surv = numpyro.deterministic("survival", sim["survival"])

    # indexing
    substance_idx = obs["substance_index"]
    sigma_cint_indexed = sigma_cint[substance_idx]
    sigma_nrf2_indexed = sigma_nrf2[substance_idx]

    sigma_cint_ix_bc = jnp.broadcast_to(sigma_cint_indexed.reshape((-1, 1)), obs["cint"].shape)
    sigma_nrf2_ix_bc = jnp.broadcast_to(sigma_nrf2_indexed.reshape((-1, 1)), obs["nrf2"].shape)

    # error model
    S = jnp.clip(surv, EPS, 1 - EPS) 
    S_cond = S[:, 1:] / S[:, :-1]
    S_cond_ = jnp.column_stack([jnp.ones_like(substance_idx), S_cond])

    n_surv = obs["survivors_before_t"]
    S_mask = masks["survival"]
    obs_survival = obs["survival"]
    
    numpyro.sample("cint_obs", dist.LogNormal(loc=jnp.log(cint + EPS), scale=sigma_cint_ix_bc).mask(masks["cint"]), obs=obs["cint"])
    numpyro.sample("nrf2_obs", dist.LogNormal(loc=jnp.log(nrf2), scale=sigma_nrf2_ix_bc).mask(masks["nrf2"]), obs=obs["nrf2"])    
    numpyro.sample(
        "survival_obs", 
        dist.Binomial(probs=S_cond_, total_count=n_surv).mask(S_mask), 
        obs=obs_survival
    )



# RNA pulse model 3.6d 
# ====================
# removes the fit on survival data

def model_rna_pulse_3_6d_substance_independent_rna_protein_module(solver, obs, masks, ci_max, only_prior=False):
    """Probability model with substance specific parameters and a conditional 
    binomial probability model for survival.
    
    Description of the probability model
    ------------------------------------
    The model is written so that MCMC samples are drawn from standard normal
    distributions, which are then deterministically mapped onto the log-normal
    space. This is helpful for using stochastic variational inference (SVI) with
    a multivariate normal target distribution. This target distribution can then
    in theory be used as a sampling distribution to generate proposals for true
    MCMC with NUTS.

    The model than inputs the parameters into the ODE solver, which maps the 
    parameters onto the different experiments with the different substances. And
    calculates ODE solutions for all experiments entered into the Simulation.
    Finally the log-probabilities of the observations (masked to exclude missing
    observations) are compared given the solutions of the ODE and the sigmas
    of the error model.

    Model parameters
    ----------------
    k_i: uptake rate
    r_rt: gene expression rate
    r_rd: RNA decay rate
    v_rt: Speed of on-switch of gene expression
    z_ci: internal concentration threshold for NRF2 activation 
    r_pt: Metabolization protein translation rate from RNA expression
    r_pd: Protein decay rate 
    sigma_*: scale parameter of the error-distributions.

    Dependency structure
    --------------------
    The following parameters are assumed as independent w.r.t to the substances:
    - r_rt
    - r_rd
    - k_p
    - v_rt
    - sigma_cint
    - sigma_nrf2
    """
    EPS = 1e-10

    # substance specific parameters
    k_i    = lognormal_prior(name="k_i"   , scale=2, loc=[1.0  , 1.0 , 1.0   ], normal_base=True)
    k_m    = lognormal_prior(name="k_m"   , scale=2, loc=[0.05 , 0.05, 0.05  ], normal_base=True)
    z_ci   = lognormal_prior(name="z_ci"  , scale=2, loc=[0.5  , 0.5 , 0.5   ], normal_base=True)
        
    # general parameters
    r_rt   = lognormal_prior(name="r_rt"  , scale=2, loc=[1.0  ], normal_base=True)
    r_rd   = lognormal_prior(name="r_rd"  , scale=2, loc=[0.5  ], normal_base=True)
    k_p    = lognormal_prior(name="k_p"   , scale=2, loc=[0.02 ], normal_base=True)
    v_rt   = lognormal_prior(name="v_rt"  , scale=2, loc=[10   ], normal_base=True)

    # error model sigmas
    sigma_cint = halfnormal_prior(name="sigma_cint", loc=0, scale=[5.0 ], normal_base=True)
    sigma_nrf2 = halfnormal_prior(name="sigma_nrf2", loc=0, scale=[5.0 ], normal_base=True)

    ci_max_ = numpyro.deterministic("ci_max", ci_max)

    if only_prior:
        # stop early if only the prior should be sampled
        return

    # deterministic model
    theta = {
        "k_i": k_i,
        "k_m": k_m,
        "r_rt": r_rt,
        "r_rd": r_rd,
        "v_rt": v_rt,
        "z_ci": z_ci,
        # Ci_max must have the same order as ot
        "ci_max": ci_max_,
        "k_p": k_p,
    }

    sim = solver(theta=theta)
    cext = numpyro.deterministic("cext", sim["cext"])
    cint = numpyro.deterministic("cint", sim["cint"])
    P = numpyro.deterministic("P", sim["P"])
    nrf2 = numpyro.deterministic("nrf2", sim["nrf2"])

    # indexing
    substance_idx = obs["substance_index"]
    sigma_cint_indexed = sigma_cint[substance_idx]
    sigma_nrf2_indexed = sigma_nrf2[substance_idx]

    sigma_cint_ix_bc = jnp.broadcast_to(sigma_cint_indexed.reshape((-1, 1)), obs["cint"].shape)
    sigma_nrf2_ix_bc = jnp.broadcast_to(sigma_nrf2_indexed.reshape((-1, 1)), obs["nrf2"].shape)

    # error model
    numpyro.sample("cint_obs", dist.LogNormal(loc=jnp.log(cint + EPS), scale=sigma_cint_ix_bc).mask(masks["cint"]), obs=obs["cint"])
    numpyro.sample("nrf2_obs", dist.LogNormal(loc=jnp.log(nrf2), scale=sigma_nrf2_ix_bc).mask(masks["nrf2"]), obs=obs["nrf2"])    

