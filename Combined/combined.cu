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

__global__ void cfarKernelShared(const float* input, int* detections, unsigned int numCells) 
{
	// Define parameters for CFAR
	const int numTrain = 5;   // Number of training cells
	const int numGuard = 1;    // Number of guard cells
	const float rateFA = 2; // Desired false alarm rate (dependent on input data)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

	//allocate shared memory
	extern __shared__ float shared_input[];

	//clear shared memory array
	//shared_input[threadIdx.x] = 0.0;

	// Bring in input data to local shared_input array
	if (tid < numCells)
	{
		shared_input[threadIdx.x] = input[tid];
	}
	__syncthreads();

    // Calculate the average for the lagging window
    float sumTrain = 0.0f;
    for (int i = threadIdx.x - numTrain; i <= threadIdx.x - numGuard; ++i) 
	{
        if (i >= 0 && i < 256) 
		{
            sumTrain += shared_input[i];
        }
    }
	
	// Calculate the average for the leading window
	for (int i = threadIdx.x + numTrain; i >= threadIdx.x + numGuard; --i)
	{
		if (i >= 0 && i < 256)
		{
			sumTrain += shared_input[i];
		}
	}
	
	// Find the average of the lagging and leading windows
    float threshold = (sumTrain / (numTrain * 2));
	__syncthreads();

	// If the current cell is above the local average, target detected
    if (shared_input[threadIdx.x] > threshold * rateFA) 
	{
        detections[tid] = 1; // Detected target
    } 
	else 
	{
        detections[tid] = 0; // No target
    }
}


void checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << message << " - " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Placeholder for measurements
    std::vector<float> measurements(1024, 0.5); // Example data

    int n = measurements.size();
    std::vector<float> stateEstimate(n, 0);
    std::vector<float> covarianceEstimate(n, 1);
    std::vector<int> detections(n, 0);

    float *d_stateEstimate, *d_covarianceEstimate, *d_measurements;
    int *d_detections;

    checkCudaError(cudaMalloc(&d_stateEstimate, n * sizeof(float)));
    checkCudaError(cudaMalloc(&d_covarianceEstimate, n * sizeof(float)));
    checkCudaError(cudaMalloc(&d_measurements, n * sizeof(float)));
    checkCudaError(cudaMalloc(&d_detections, n * sizeof(int)));

    checkCudaError(cudaMemcpy(d_stateEstimate, stateEstimate.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_covarianceEstimate, covarianceEstimate.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_measurements, measurements.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    float processNoise = 1e-5;
    float measurementNoise = 1e-5;

    cudaStream_t stream;
    checkCudaError(cudaStreamCreate(&stream));

    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start));
    checkCudaError(cudaEventCreate(&stop));

    checkCudaError(cudaEventRecord(start, stream));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    int sharedMemSize = 3 * threadsPerBlock * sizeof(float);

    kalmanFilter<<<blocksPerGrid, threadsPerBlock, sharedMemSize, stream>>>(d_stateEstimate, d_covarianceEstimate, d_measurements, processNoise, measurementNoise, n);
    cfarKernelShared<<<blocksPerGrid, threadsPerBlock, sharedMemSize, stream>>>(d_stateEstimate, d_detections, n);

    checkCudaError(cudaEventRecord(stop, stream));
    checkCudaError(cudaEventSynchronize(stop));

    float milliseconds = 0;
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Total kernel execution time: " << milliseconds << " milliseconds\n";

    checkCudaError(cudaMemcpyAsync(stateEstimate.data(), d_stateEstimate, n * sizeof(float), cudaMemcpyDeviceToHost, stream));
    checkCudaError(cudaMemcpyAsync(covarianceEstimate.data(), d_covarianceEstimate, n * sizeof(float), cudaMemcpyDeviceToHost, stream));
    checkCudaError(cudaMemcpyAsync(detections.data(), d_detections, n * sizeof(int), cudaMemcpyDeviceToHost, stream));

    checkCudaError(cudaStreamSynchronize(stream));

    checkCudaError(cudaFree(d_stateEstimate));
    checkCudaError(cudaFree(d_covarianceEstimate));
    checkCudaError(cudaFree(d_measurements));
    checkCudaError(cudaFree(d_detections));

    checkCudaError(cudaStreamDestroy(stream));
    checkCudaError(cudaEventDestroy(start));
    checkCudaError(cudaEventDestroy(stop));

    return 0;
}