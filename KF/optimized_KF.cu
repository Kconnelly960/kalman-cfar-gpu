#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cuda_runtime.h>

__global__ void kalmanFilter(float *stateEstimates, float *covariances, const float *measurements, float processNoise, float measurementNoise, int numElements) {
    extern __shared__ float sharedData[]; // Declare dynamically allocated shared memory

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int localIdx = threadIdx.x; // Index in the block

    // Load data into shared memory
    if (idx < numElements) {
        sharedData[localIdx] = stateEstimates[idx];
        sharedData[localIdx + blockDim.x] = covariances[idx];
        sharedData[localIdx + 2 * blockDim.x] = measurements[idx];
    }
    __syncthreads(); // Ensure all data is loaded into shared memory before proceeding

    if (idx < numElements) {
        // Predict
        float predictedState = sharedData[localIdx];
        float predictedCovariance = sharedData[localIdx + blockDim.x] + processNoise;

        // Update
        float innovation = sharedData[localIdx + 2 * blockDim.x] - predictedState;
        float innovationCovariance = predictedCovariance + measurementNoise;
        float kalmanGain = predictedCovariance / innovationCovariance;

        sharedData[localIdx] = predictedState + kalmanGain * innovation; // Updated state estimate
        sharedData[localIdx + blockDim.x] = (1 - kalmanGain) * predictedCovariance; // Updated covariance estimate
    }
    __syncthreads(); // Ensure all calculations are completed before writing back to global memory

    // Write updated estimates back to global memory
    if (idx < numElements) {
        stateEstimates[idx] = sharedData[localIdx];
        covariances[idx] = sharedData[localIdx + blockDim.x];
    }
}


void checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << message << " - " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}


int main() {
    std::string line;
    std::ifstream file("./radarDataMatlab.csv");
    std::vector<float> measurements;

    if (file.is_open()) {
        while (getline(file, line)) {
            std::stringstream linestream(line);
            std::string cell;
            getline(linestream, cell, ',');  // Skip time column
            getline(linestream, cell, ',');
            measurements.push_back(std::stof(cell));
        }
        file.close();
    } else {
        std::cerr << "Unable to open file" << std::endl;
        return EXIT_FAILURE;
    }

    int n = measurements.size();
    std::vector<float> stateEstimate(n, 0);  // Initial state estimates
    std::vector<float> covarianceEstimate(n, 1);  // Initial covariance estimates

    float *d_stateEstimate, *d_covarianceEstimate, *d_measurements;
    checkCudaError(cudaMalloc(&d_stateEstimate, n * sizeof(float)), "allocating d_stateEstimate");
    checkCudaError(cudaMalloc(&d_covarianceEstimate, n * sizeof(float)), "allocating d_covarianceEstimate");
    checkCudaError(cudaMalloc(&d_measurements, n * sizeof(float)), "allocating d_measurements");

    checkCudaError(cudaMemcpy(d_stateEstimate, stateEstimate.data(), n * sizeof(float), cudaMemcpyHostToDevice), "copying to d_stateEstimate");
    checkCudaError(cudaMemcpy(d_covarianceEstimate, covarianceEstimate.data(), n * sizeof(float), cudaMemcpyHostToDevice), "copying to d_covarianceEstimate");
    checkCudaError(cudaMemcpy(d_measurements, measurements.data(), n * sizeof(float), cudaMemcpyHostToDevice), "copying to d_measurements");

    float processNoise = 1e-5;  // Example process noise
    float measurementNoise = 1e-5;  // Example measurement noise

    // Set up CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start recording
    cudaEventRecord(start);

    // Launch the kernel
    int threadsPerBlock = 8;
    int blocksPerGrid = 512;
    size_t sharedMemSize = 3 * threadsPerBlock * sizeof(float); // Allocating for stateEstimates, covariances, and measurements

    kalmanFilter<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_stateEstimate, d_covarianceEstimate, d_measurements, processNoise, measurementNoise, n);

    // Stop recording
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate and print the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " milliseconds\n and the number of blocks is " << blocksPerGrid << " and threads is" << threadsPerBlock << "\n";

    checkCudaError(cudaMemcpy(stateEstimate.data(), d_stateEstimate, n * sizeof(float), cudaMemcpyDeviceToHost), "copying from d_stateEstimate");
    checkCudaError(cudaMemcpy(covarianceEstimate.data(), d_covarianceEstimate, n * sizeof(float), cudaMemcpyDeviceToHost), "copying from d_covarianceEstimate");

    cudaFree(d_stateEstimate);
    cudaFree(d_covarianceEstimate);
    cudaFree(d_measurements);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}