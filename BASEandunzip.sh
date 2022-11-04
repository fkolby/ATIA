#!/bin/bash
#SBATCH --job-name=Unzip
#number of independent tasks we are going to start in this script
#SBATCH --ntasks=1 --cpus-per-task=4
#SBATCH -o OUT-%x-%j.txt
#We expect that our program should not run longer than .5 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=00:30:00


#ensure that batchsize X nodes X devices=64
#your script, in this case: write the hostname and the ids of the chosen gpus.
unzip chest-xray-pneumonia.zip  