
# Function to generate input paths based on input tuple index
def get_input_likelihood(wildcards):
    # Access tuple values using the index from your wildcards
    # Construct and return the necessary list/dictionary of input files
    return {
        "config": f"scenarios/{wildcards.scenario}/settings.cfg",
        "posterior": f"results/{wildcards.scenario}/numpyro_posterior.nc",
    }


rule get_input_likelihood:
    input:
        unpack(get_input_files)

    output:
        "results/{scenario}/likelihood_landscapes/parx~{parx}__pary~{pary}.png"

    shell: """
        plot-likelihood-landscape \
            --config={input.config} \
            --parx={wildcards.parx} \
            --pary={wildcards.pary} \
            --std_dev={config[likelihood_landscapes][std_dev]} \
            --n_grid_points={config[likelihood_landscapes][n_grid_points]} \
            --n_vector_points={config[likelihood_landscapes][n_vector_points]} \
            --idata_file="numpyro_posterior.nc" \
            --sparse \
            --center-at-posterior-mode
    """
