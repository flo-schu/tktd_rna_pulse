import pytest
from tktd_rna_pulse.sim import SingleSubstanceSim3


def construct_sim(scenario, simulation_class):
    """Helper function to construct simulations for debugging"""
    sim = simulation_class(f"scenarios/{scenario}/settings.cfg")
    sim.setup()
    return sim


# List test scenarios and simulations
@pytest.fixture(scope="module", params=[
    "rna_pulse_3_6c_substance_specific",
    "rna_pulse_3_6c_substance_independent_rna_protein_module"
])
def scenario(request):
    return request.param

@pytest.fixture(scope="module", params=[
    SingleSubstanceSim3
])
def simulation_class(request):
    return request.param


# Derive simulations for testing from fixtures
@pytest.fixture(scope="module")
def sim(scenario, simulation_class):
    yield construct_sim(scenario, simulation_class)


# run tests with the Simulation fixtures
def test_setup(sim):
    """Tests the construction method"""
    assert True


def test_simulation(sim):
    """Tests if a forward simulation pass can be computed"""
    sim.dispatch_constructor()
    evaluator = sim.dispatch({})
    evaluator()
    evaluator.results

    assert True
            

@pytest.mark.parametrize("backend", ["numpyro"])
def test_inference(sim, backend):
    """Tests if prior predictions can be computed for arbitrary backends"""
    sim.dispatch_constructor()
    sim.set_inferer(backend)

    # The below does not work with the new backend, because the priors
    # are sampled in the substance dimension, but need to be mapped to the
    # id dimension. This can be done manually, and in complex cases this is
    # required, but for many cases something like:
    # map_to=("id",) should be fine. By default the index for mapping substance
    # to ID is called substance_index. As the dimension of the variable
    # is substance. This can be filled in automatically. The big advantage of 
    # this is that it can be done in the evaluator, if the keyword is set.
    sim.inferer.prior_predictions(n=2)



if __name__ == "__main__":
    import os
    if os.path.basename(os.getcwd()) != "tktd_rna_pulse":
        # change directory to case_studies/beeguts
        # this may work in case the root is a project with case_studies
        os.chdir("case_studies/tktd_rna_pulse")

    test_simulation(sim=construct_sim("rna_pulse_3_6c_substance_specific", SingleSubstanceSim3))