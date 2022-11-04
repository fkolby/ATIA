#!/bin/bash
#SBATCH --job-name=Patch
#number of independent tasks we are going to start in this script
#SBATCH --ntasks=1 --cpus-per-task=12 -p gpu --gres=gpu:titanx:1 --mem=6000M
#SBATCH --array=1-10%3
#SBATCH -o OUT-%x-%j-%a.txt
#We expect that our program should not run longer than 2.5 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=03:00:00


#ensure that batchsize X nodes X devices=64
#your script, in this case: write the hostname and the ids of the chosen gpus.
hostname
echo $CUDA_VISIBLE_DEVICES
python ~/ATIAjbv/train/ATIApyfilemain.py --seed $SLURM_ARRAY_TASK_ID --modeltype Patch --max_epochs 100 --testSetName test --maxTime 10800 --gpu True --devices 1 --workers 10 --batchSize 64
