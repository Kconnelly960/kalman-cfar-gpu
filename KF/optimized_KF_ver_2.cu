#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cuda_runtime.h>

__global__ void kalmanFilter(float *stateEstimates, float *covariances, const float *measurements, float processNoise, float measurementNoise, int numElements) {
    extern __shared__ float sharedData[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int localIdx = threadIdx.x;

    if (idx < numElements) {
        sharedData[localIdx] = stateEstimates[idx];
        sharedData[localIdx + blockDim.x] = covariances[idx];
        sharedData[localIdx + 2 * blockDim.x] = measurements[idx];
    }

    __syncthreads();

    if (idx < numElements) {
        float predictedState = sharedData[localIdx];
        float predictedCovariance = sharedData[localIdx + blockDim.x] + processNoise;
        float innovation = sharedData[localIdx + 2 * blockDim.x] - predictedState;
        float innovationCovariance = predictedCovariance + measurementNoise;
        float kalmanGain = predictedCovariance / innovationCovariance;

        sharedData[localIdx] = predictedState + kalmanGain * innovation;
        sharedData[localIdx + blockDim.x] = (1 - kalmanGain) * predictedCovariance;
    }

    __syncthreads();

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
    std::vector<float> measurements; // Assume this is populated from file reading

    int n = measurements.size();
    std::vector<float> stateEstimate(n, 0);
    std::vector<float> covarianceEstimate(n, 1);

    float *d_stateEstimate, *d_covarianceEstimate, *d_measurements;
    cudaMalloc(&d_stateEstimate, n * sizeof(float));
    cudaMalloc(&d_covarianceEstimate, n * sizeof(float));
    cudaMalloc(&d_measurements, n * sizeof(float));

    cudaMemcpy(d_stateEstimate, stateEstimate.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_covarianceEstimate, covarianceEstimate.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_measurements, measurements.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    float processNoise = 1e-5;
    float measurementNoise = 1e-5;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);

    int threadsPerBlock = 256; // Adjusted as per device capability
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    int sharedMemSize = 3 * threadsPerBlock * sizeof(float);

    kalmanFilter<<<blocksPerGrid, threadsPerBlock, sharedMemSize, stream>>>(d_stateEstimate, d_covarianceEstimate, d_measurements, processNoise, measurementNoise, n);

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " milliseconds\n";

    cudaMemcpyAsync(stateEstimate.data(), d_stateEstimate, n * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(covarianceEstimate.data(), d_covarianceEstimate, n * sizeof(float), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    cudaFree(d_stateEstimate);
    cudaFree(d_covarianceEstimate);
    cudaFree(d_measurements);

    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
