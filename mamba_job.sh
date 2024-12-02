#!/bin/sh 
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J mamba_job_gpu
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 2 gpus in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 16GB of system-memory
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "select[gpu32gb]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s242830@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --

nvidia-smi
# Activate venv
module load python3/3.10.12
module load cuda/11.8
module load cudnn/v8.8.0-prod-cuda-11.X
source /zhome/0d/6/213435/Structured_SSM_for_EHR_Classification/env/bin/activate

# run training
python cli.py --output_path=/zhome/0d/6/213435/Structured_SSM_for_EHR_Classification_Group38/mamba_results --epochs=100 --batch_size=64 --model_type=mamba --dropout=0.2 --attn_dropout=0.1 --layers=3 --heads=1 --pooling=max --lr=0.0001