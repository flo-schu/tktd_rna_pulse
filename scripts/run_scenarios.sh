#!/usr/bin/env bash
#SBATCH --job-name=hierarchical-molecular-tktd          # name of job
#SBATCH --time=0-12:00:00                               # maximum time until job is cancelled
#SBATCH --ntasks=1                                      # number of tasks
#SBATCH --cpus-per-task=1                               # number of cpus requested
#SBATCH --mem-per-cpu=32G                               # memory per cpu requested
#SBATCH --mail-type=begin                               # send mail when job begins
#SBATCH --mail-type=end                                 # send mail when job ends
#SBATCH --mail-type=fail                                # send mail if job fails
#SBATCH --mail-user=florian.schunck@uos.de              # email of user
#SBATCH --output=/home/staff/f/fschunck/logs/job-%x-%A_%a.out    # output file of stdout messages
#SBATCH --error=/home/staff/f/fschunck/logs/job-%x-%A_%a.err     # output file of stderr messages


echo "starting job"

# Define a manually specified list of inputs
scenarios=(
    #"hierarchical_cext_nested_sigma_hyperprior"
    "hierarchical_cext_nested_sigma_hyperprior_reduced_dataset"
)

# Get the index for the current array job
index=$SLURM_ARRAY_TASK_ID

# Access the corresponding input from the list
input_scenario=${scenarios[$index]}

spack load miniconda3
source activate hmt
spack unload miniconda3

export JAX_ENABLE_X64=True
export XLA_FLAGS="--xla_force_host_platform_device_count=${SLURM_CPUS_PER_TASK}"

srun pymob-infer \
    --case_study=hierarchical_molecular_tktd \
    --scenario=$input_scenario \
    --package=.. \
    --n_cores $SLURM_CPUS_PER_TASK \
    --inference_backend=numpyro

echo "Finished job."
