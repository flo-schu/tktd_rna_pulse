
# Function to generate input paths based on input tuple index
def get_input_files(wildcards):
    # Access tuple values using the index from your wildcards
    # Construct and return the necessary list/dictionary of input files
    return {
        "config": f"scenarios/{wildcards.scenario}/settings.cfg",
    }


rule pymob_infer:
    input:
        unpack(get_input_files)
    output:
        "results/{scenario}/out.txt", 
        "results/{scenario}/log.txt", 
        "results/{scenario}/numpyro_posterior.nc", 
        "results/{scenario}/combined_pps_figure_diclofenac.png", 
        "results/{scenario}/prior_predictive.png", 
        "results/{scenario}/posterior_predictive.png", 
        "results/{scenario}/combined_pps_figure_diuron.png", 
        "results/{scenario}/combined_pps_figure_naproxen.png", 
        "results/{scenario}/pairs_posterior.png", 
        "results/{scenario}/trace.png", 
        "results/{scenario}/svi_loss_curve.png", 
        "results/{scenario}/settings.cfg", 
        "results/{scenario}/probability_model.png", 
        "results/{scenario}/parameter_pairs_likelihood_landscape.tsv"


    # TODO: Integrate datalad :) unlock, compute save. Write the last commit 
    # id into the commit message for reproducibility.
    shell: """
        echo "Running Workflow for {input.config}" > {output}
        pymob-infer \
            --case_study=tktd_rna_pulse \
            --scenario={wildcards.scenario} \
            --package=.. \
            --n_cores 1 \
            --inference_backend=numpyro
        """
