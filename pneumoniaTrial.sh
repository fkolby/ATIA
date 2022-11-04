#!/bin/bash
#SBATCH --job-name=BaseCaseTrial
#number of independent tasks we are going to start in this script
#SBATCH --ntasks=1 --cpus-per-task=12 -p gpu --gres=gpu:titanx:1 --mem=6000M
#SBATCH -o OUT-%x-%j.txt
#We expect that our program should not run longer than 2.5 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=00:10:00


#ensure that batchsize X nodes X devices=64
#your script, in this case: write the hostname and the ids of the chosen gpus.
hostname
echo $CUDA_VISIBLE_DEVICES
python /home/jbv415/ATIAjbv/Pneumonia/chestXray/chest_xray/pneumoniaATIApyfilemain.py --limitTrain 0.25 --seed 1 --max_epochs 5 --testSetName test --maxTime 300 --gpu True --devices 1 --workers 10 --batchSize 64
