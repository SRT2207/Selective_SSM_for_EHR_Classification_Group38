#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J mamba_job_cpu
### -- ask for number of cores (default: 1) -- 
#BSUB -n 16 
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 64GB of memory per core/slot -- 
#BSUB -R "rusage[mem=128GB]"
### -- specify that we want the job to get killed if it exceeds 10 GB per core/slot -- 
#BSUB -M 130GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Output_%J.out 
#BSUB -e Output_%J.err 

# here follow the commands you want to execute with input.in as the input file
module load python3/3.10.12
source /zhome/0d/6/213435/Structured_SSM_for_EHR_Classification/env/bin/activate

python cli.py --output_path=/zhome/0d/6/213435/Structured_SSM_for_EHR_Classification_Group38/mamba_results --epochs=100 --batch_size=16 --model_type=mamba --dropout=0.2 --attn_dropout=0.1 --layers=3 --heads=1 --pooling=max --lr=0.0001