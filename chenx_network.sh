#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --mem=500mb
#SBATCH -t 12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chen7090@umn.edu
#SBATCH -p small
#SBATCH -o %j_network.out
#SBATCH -e %j_network.err

hostname

module load python3/3.8.3_anaconda2020.07_mamba
module load intel 

echo "Module loaded. This is to use SWMM without multiprocessing."
folder_str='./datafiles_pool_'
dt_str=$(date +'%Y%m%d-%H%M')
folder_name="${folder_str}${dt_str}_${SLURM_JOBID}"
echo $folder_name
mkdir $folder_name
cd $folder_name
echo "walltime=12:00:00,nodes=1:ppn=3,pmem=500mb,-p small,no np"
echo "flood level 10, range(10)"
echo "this version does not have split soil_node_range"
echo "running now"
python3 ../make_SWMM_inp.py