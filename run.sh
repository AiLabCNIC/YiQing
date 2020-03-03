#!/bin/bash
#SBATCH -p cuda10-ceshi
#SBATCH -N 1
#SBATCH --gres=gpu:8

python main.py