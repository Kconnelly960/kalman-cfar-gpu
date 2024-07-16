#!/bin/bash

#SBATCH --job-name=basic_kalman
#SBATCH --acount=ece569
#SBATCH --partition=standard
#SBATCH -e run.error
#SBATCH -o run.out

### Set the number of nodes, cores and memory that will be used for this job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=12gb
### Specify "wallclock time", hhh:mm:ss. Required field
#SBATCH --time 00:00:05

### Load required modules/libraries if needed
### Assuming cuda module is available
module load cuda11/11.0

### set directory for job execution, replace the following path with your own path
hw_path=/home/u8/kconnelly1/ece569/project

cd $hw_path


################################################################################################################################
################################################################################################################################

## BASIC KALMAN FILTER IMPLEMENTATION ##

### Compile the basic kalman filter CUDA program
nvcc -o ./KF/basic_implementation_kalman_filter ./KF/basic_implementation_kalman_filter.cu

### Initialize an array to hold execution times
declare -a times_a

### Run the CUDA binary 30 times and capture the execution time
for i in {1..30}
do
   # Run the program and use grep to extract the execution time from output
   nvprof --log-file ./KF/basic_KF_results/${i}_profile.txt ./KF/basic_implementation_kalman_filter > ./KF/basic_KF_results/${i}_run_output.txt
   
   # Use grep to extract the execution time from output
   time=$(grep -oP 'Kernel execution time: \K[0-9.]+(?= milliseconds)' ./KF/basic_KF_results/${i}_run_output.txt)
   times_a+=($time)
   rm ./KF/basic_KF_results/${i}_run_output.txt
done

# Calculate the average execution time
total=0
for t in "${times_a[@]}"
do
    total=$(echo "$total + $t" | bc)
done
average=$(echo "scale=2; $total / 30" | bc)

# Output the average execution time and all times to performance_metrics.txt
echo "Average Kernel Execution Time: $average milliseconds" > ./KF/basic_KF_results/performance_metrics.txt
echo "Detailed Kernel Execution Times:" >> ./KF/basic_KF_results/performance_metrics.txt
for t in "${times_a[@]}"
do
    echo "$t milliseconds" >> ./KF/basic_KF_results/performance_metrics.txt
done


################################################################################################################################
################################################################################################################################


## STREAMING KALMAN FILTER ##

### Compile the second optimizations version
nvcc -o ./KF/optimized_KF_ver_2 ./KF/optimized_KF_ver_2.cu

### Initialize an array to hold execution times
declare -a times_b

### Run the CUDA binary 30 times and capture the execution time
for i in {1..30}
do
   # Run the program and use grep to extract the execution time from output
   nvprof --log-file ./KF/optimized_KF_ver_2_results/${i}_profile.txt ./KF/optimized_KF_ver_2 > ./KF/optimized_KF_ver_2_results/${i}_run_output.txt
   
   # Use grep to extract the execution time from output
   time=$(grep -oP 'Kernel execution time: \K[0-9.]+(?= milliseconds)' ./KF/optimized_KF_ver_2_results/${i}_run_output.txt)
   times_b+=($time)
   rm ./KF/optimized_KF_ver_2_results/${i}_run_output.txt
done

# Calculate the average execution time
total=0
for t in "${times_b[@]}"
do
    total=$(echo "$total + $t" | bc)
done
average=$(echo "scale=2; $total / 30" | bc)

# Output the average execution time and all times to performance_metrics.txt
echo "Average Kernel Execution Time: $average milliseconds" > ./KF/optimized_KF_ver_2_results/performance_metrics.txt
echo "Detailed Kernel Execution Times:" >> ./KF/optimized_KF_ver_2_results/performance_metrics.txt
for t in "${times_b[@]}"
do
    echo "$t milliseconds" >> ./KF/optimized_KF_ver_2_results/performance_metrics.txt
done


################################################################################################################################
################################################################################################################################

## SHARED MEMORY KALMAN FILTER ##

### Compile the optimized kalman filter CUDA program that includes shared memory
nvcc -o ./KF/optimized_KF ./KF/optimized_KF.cu


### Initialize an array to hold execution times
declare -a times_c

### Run the CUDA binary 30 times and capture the execution time
for i in {1..30}
do
   # Run the program and use grep to extract the execution time from output
   nvprof --log-file ./KF/optimized_KF_results/${i}_profile.txt ./KF/optimized_KF > ./KF/optimized_KF_results/${i}_run_output.txt
   
   # Use grep to extract the execution time from output
   time=$(grep -oP 'Kernel execution time: \K[0-9.]+(?= milliseconds)' ./KF/optimized_KF_results/${i}_run_output.txt)
   times_c+=($time)
   rm ./KF/optimized_KF_results/${i}_run_output.txt
done

# Calculate the average execution time
total=0
for t in "${times_c[@]}"
do
    total=$(echo "$total + $t" | bc)
done
average=$(echo "scale=2; $total / 30" | bc)

# Output the average execution time and all times to performance_metrics.txt
echo "Average Kernel Execution Time: $average milliseconds" > ./KF/optimized_KF_results/performance_metrics.txt
echo "Detailed Kernel Execution Times:" >> ./KF/optimized_KF_results/performance_metrics.txt
for t in "${times_c[@]}"
do
    echo "$t milliseconds" >> ./KF/optimized_KF_results/performance_metrics.txt
done


################################################################################################################################
################################################################################################################################

## COMBINED PROJECT ##

### Compile the combined CUDA program that includes shared memory
nvcc -o ./Combined/combined ./Combined/combined.cu


### Initialize an array to hold execution times
declare -a times_d

### Run the CUDA binary 30 times and capture the execution time
for i in {1..30}
do
   # Run the program and use grep to extract the execution time from output
   nvprof --log-file ./Combined/combined_results/${i}_profile.txt ./Combined/combined > ./Combined/combined_results/${i}_run_output.txt
   
   # Use grep to extract the execution time from output
   time=$(grep -oP 'Kernel execution time: \K[0-9.]+(?= milliseconds)' ./Combined/combined_results/${i}_run_output.txt)
   times_c+=($time)
   rm ./Combined/combined_results/${i}_run_output.txt
done

# Calculate the average execution time
total=0
for t in "${times_c[@]}"
do
    total=$(echo "$total + $t" | bc)
done
average=$(echo "scale=2; $total / 30" | bc)

# Output the average execution time and all times to performance_metrics.txt
echo "Average Kernel Execution Time: $average milliseconds" > ./Combined/combined_results/performance_metrics.txt
echo "Detailed Kernel Execution Times:" >> ./Combined/combined_results/performance_metrics.txt
for t in "${times_c[@]}"
do
    echo "$t milliseconds" >> ./Combined/combined_results/performance_metrics.txt
done


################################################################################################################################
################################################################################################################################


