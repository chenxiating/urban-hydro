#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=1000mb
#SBATCH -t 24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chen7090@umn.edu
#SBATCH -p small
#SBATCH -o %j_network.out
#SBATCH -e %j_network.err

hostname

module load python3/3.8.3_anaconda2020.07_mamba
module load intel 

echo "This is to generate Gibbs network."
echo "walltime=24:00:00,nodes=1:ppn=8,pmem=1000mb,-p small,no np"
python3 run_mp_gibbs.py
