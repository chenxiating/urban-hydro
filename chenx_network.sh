#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --mem=500mb
#SBATCH -t 24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chen7090@umn.edu
#SBATCH -p small
#SBATCH -o %j_network.out
#SBATCH -e %j_network.err

hostname

module load python3/3.8.3_anaconda2020.07_mamba
module load intel 

echo "module loaded"
folder_str='./datafiles_pool_'
dt_str=$(date +'%Y%m%d-%H%M')
folder_name="${folder_str}${dt_str}_${SLURM_JOBID}"
echo $folder_name
mkdir $folder_name
echo "walltime=24:00:00,nodes=1:ppn=24,pmem=500mb,np -1"
echo "running multiprocessing now"
mpirun -np 1 python3 run_multiprocessing_pool.py $dt_str
mv dataset*$dt_str* $folder_name
