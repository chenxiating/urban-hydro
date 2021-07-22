#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1000mb
#SBATCH -t 16:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chen7090@umn.edu
#SBATCH -p small
#SBATCH -o %j_network.out
#SBATCH -e %j_network.err

hostname

module load python3/3.8.3_anaconda2020.07_mamba
module load intel 

echo "Module loaded. This is to use SWMM with multiprocessing."
echo "walltime=8:00:00,nodes=1:ppn=5,pmem=1000mb,-p small,no np"
echo "this version has multiprocessing"
echo "running now"
python3 run_multiprocessing_pool.py
