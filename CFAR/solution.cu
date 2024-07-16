#include <cuda_runtime.h>
#include<stdlib.h>
#include <wb.h>
#include <iostream>

#include "kernel.cu"


int main() 
{
	int debug = 0;
	int version = 0;
	const int numCells = 256 * 1024 * 20; // Number of cells in the range profile

	cudaEvent_t astartEvent, astopEvent;
	float aelapsedTime;
	cudaEventCreate(&astartEvent);
	cudaEventCreate(&astopEvent);

    // Generate synthetic input data for testing

    float* inputData = new float[numCells];

	for (int i = 0; i < numCells; ++i)
	{
		inputData[i] = 20 + std::rand() % 20;
	}
	inputData[1152] = 90;
	inputData[1153] = 100;
	inputData[1154] = 90;

	if (debug == 1) {
	for (int j = 0; j < 10; ++j)
	{
		printf("inputdata[%d]: %f \n", j, inputData[j]);
	} }

    // Allocate memory on the GPU
    float* d_inputData;
    int* d_detections;
	int* d_sharedDetections;
    cudaMalloc(&d_inputData, numCells * sizeof(float));
    cudaMalloc(&d_detections, numCells * sizeof(int));
    cudaMalloc(&d_sharedDetections, numCells * sizeof(int));

    // Copy input data to the GPU
    cudaMemcpy(d_inputData, inputData, numCells * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the CFAR kernel
    const unsigned int blockSize = 256;
    const unsigned int numBlocks = (numCells + blockSize - 1) / blockSize;

	//*****************Naive Version***********************
	if (version == 0) {
  	cudaEventRecord(astartEvent, 0);
    cfarKernel<<<numBlocks, blockSize>>>(d_inputData, d_detections, numCells);

	cudaEventRecord(astopEvent, 0);
	cudaEventSynchronize(astopEvent);
	cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
	printf("\n");
	printf("Naive Version total compute time (ms) %f ",aelapsedTime);
	printf("\n"); }

	//*****************Shared Memory Version***************
	if (version == 0) {
  	cudaEventRecord(astartEvent, 0);
    cfarKernelShared<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_inputData, d_sharedDetections, numCells);

	cudaEventRecord(astopEvent, 0);
	cudaEventSynchronize(astopEvent);
	cudaEventElapsedTime(&aelapsedTime, astartEvent, astopEvent);
	printf("\n");
	printf("Shared Version total compute time (ms) %f ",aelapsedTime);
	printf("\n"); }

    // Copy results back to the CPU
    int* detections = new int[numCells];
    int* sharedDetections = new int[numCells];
    cudaMemcpy(detections, d_detections, numCells * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(sharedDetections, d_sharedDetections, numCells * sizeof(int), cudaMemcpyDeviceToHost);

	if (debug == 1) {
	printf("\n");
	for (int j = 0; j < 50; ++j)
	{
		printf("detections[%d]: %d \n", j, sharedDetections[j]);
	}
	printf("\n"); }
	if (debug == 1) 
	{
		int numDetect = 0;
		printf("\n");
		for (int i = 0; i < numCells; ++i)
		{
			if (sharedDetections[i] == 1) 
			{
				numDetect++;
				//printf("detection at: %d \n", i); 
			}

		}
		printf("Number of detections: %d \n", numDetect);
	}


    // Clean up
	cudaEventDestroy(astopEvent);
	cudaEventDestroy(astartEvent);
    delete[] inputData;
    delete[] detections;
    cudaFree(d_inputData);
    cudaFree(d_detections);
    cudaFree(d_sharedDetections);


    return 0;
}
