#!/bin/bash
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --mem=100mb
#SBATCH -t 24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chen7090@umn.edu
#SBATCH -p small
#SBATCH -o %j.out
#SBATCH -e %j.err

hostname

module load python3/3.8.3_anaconda2020.07_mamba
module load intel 
module load ompi/intel 

echo "module loaded"
folder_str='./datafiles_pool_'
dt_str=$(date +'%Y%m%d-%H%M')
folder_name="${folder_str}${dt_str}"
echo ${SLURM_JOBID}
echo $folder_name
mkdir $folder_name
echo "walltime=48:00:00,nodes=5:ppn=24,pmem=100mb"
echo "running multiprocessing now"
mpirun -np 120 python3 test_callback.py $dt_str
mv dataset*$dt_str* $folder_name
