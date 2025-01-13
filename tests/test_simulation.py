import pytest
from tktd_rna_pulse.sim import SingleSubstanceSim3

# TODO: Refactor as a test suite that works for every case study.
#       - class based. Iterates through test scenarios and options
#       - tests: Setup, Simulation, Visualization, Inference
#       - 

@pytest.fixture(scope="module", params=[
    "rna_pulse_3_6c_substance_specific",
    "rna_pulse_3_6c_substance_independent_rna_protein_module"
])
def scenario(request):
    return request.param

@pytest.fixture(scope="module", params=[SingleSubstanceSim3])
def simulation_class(request):
    return request.param

def construct_sim(scenario, simulation_class):
    sim = simulation_class(f"scenarios/{scenario}/settings.cfg")
    sim.setup()
    return sim

@pytest.fixture(scope="module")
def sim(scenario, simulation_class):
    yield construct_sim(scenario, simulation_class)

def test_setup(sim):
    # simply tests the construction method
    assert True

def test_simulation(sim):
    sim.dispatch_constructor()
    evaluator = sim.dispatch({})
    evaluator()
    evaluator.results

    # only test if no error is thrown
    assert True
            

@pytest.mark.parametrize("backend", ["numpyro"])
def test_inference(sim, backend):
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