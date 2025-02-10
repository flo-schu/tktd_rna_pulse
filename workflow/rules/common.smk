# inspired by: https://github.com/snakemake-workflows/rna-seq-star-deseq2/blob/master/workflow/rules/common.smk

import itertools
import pandas as pd
from snakemake.utils import Paramspace, validate

validate(config, schema="../schemas/config.schema.yaml")

def get_combinations(scenario):
    # Load strings from the file using pandas
    inp = f"scenarios/{scenario}/parameters_likelihood_landscape.txt"
    df = pd.read_csv(inp, header=None, names=['params'])

    # Create a list of strings from the DataFrame
    strings = df[~df['params'].str.startswith('#')]['params'].tolist()
    
    # Generate unique pairs
    pairs = list(itertools.combinations(strings, 2))

    # Create a DataFrame from the pairs
    return pd.DataFrame(pairs, columns=['parx', 'pary'])


def get_final_output():
    final_output = expand(
        "results/{scenario}/out.txt",
        scenario=config["scenarios"],
    )

    if config["likelihood_landscapes"]["run"]:
        for scenario in config["scenarios"]:
            paramspace = Paramspace(
                dataframe=get_combinations(scenario),
                filename_params=["parx", "pary"],
                filename_sep="__",
                param_sep="~",
            )
            # get all the variables to plot a PCA for
            final_output.extend(
                expand(
                    "results/{scenario}/likelihood_landscapes/{params}.png", 
                    params=paramspace.instance_patterns, scenario=scenario
                )
            )
        print(final_output)
    return final_output
