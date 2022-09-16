#!/bin/bash
#SBATCH --cpus-per-task=32
#SBATCH --sockets-per-node=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=workq
#SBATCH --time=00:10:00
###SBATCH --account=director2196
#SBATCH --account=pawsey0007

cwd=$(pwd)
module load rocm/4.5.0 gcc/11.2.0

# Fresh build of the code
make clean
CXXFLAGS="-Xcompiler -fopenmp -O2" make all

# Create a hotspot profile
srun rocprof --stats ./bin/askapDegrid.exe

# Get the kernel name of the most expensive kernel
#  >> get the first two lines of results.stats.csv
#  >> remove the header line
#  >> reverse the line, remove timing information, reverse again to give kernel name
kernel=$(head -n2 results.stats.csv | tail -n1 | rev | cut -d "," -f5- | rev)
kernel_runtime=$(head -n2 results.stats.csv | tail -n1 | rev | cut -d "," -f5- | rev)

# Get the shortened name for the kernel
kernel_short=$(echo $kernel | sed 's/void //g' | cut -d "(" -f1 | cut -d "<" -f1 | sed 's/"//g')
echo "Most expensive kernel : $kernel_short"


# Create a metrics file for rocprof from the irm template
sed "s/@KERNEL@/$kernel_short/g" rocprof-irm.tmpl > rocprof-irm.txt

srun rocprof -i rocprof-irm.txt --stats ./bin/askapDegrid.exe

# Replace the kernel name with the shortname - this makes parsing the csv easier in python for plotting
sed -i "s/$kernel/$kernel_short/g" rocprof-irm.csv


# Use performance modeling tools to get peak bandwidth and peak GIPS
git clone https://github.com/PawseySC/performance-modelling-tools 
export PERF_ODIR="${cwd}/$HOSTNAME/$(date +"%Y-%m-%d-%H-%M")"
export PERF_ROOT="${cwd}/performance-modelling-tools"
./performance-modelling-tools/benchmark/ekondis-mixbench.sh

# Move rocprof-irfm.csv output to PERF_ODIR for processing
mv rocprof-irm.csv $PERF_ODIR
