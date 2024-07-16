
// CUDA kernel for Naive CFAR
__global__ void cfarKernel(const float* input, int* detections, unsigned int numCells) 
{
	// Define parameters for CFAR
	const int numTrain = 5;   // Number of training cells
	const int numGuard = 1;    // Number of guard cells
	const float rateFA = 2; // Desired false alarm rate
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate the average for the lagging window
    float sumTrain = 0.0f;
    for (int i = tid - numTrain; i <= tid - numGuard; ++i) 
	{
        if (i >= 0 && i < numCells) 
		{
            sumTrain += input[i];
        }
    }
	
	// Calculate the average for the leading window
	for (int i = tid + numTrain; i >= tid + numGuard; --i)
	{
		if (i >= 0 && i < numCells)
		{
			sumTrain += input[i];
		}
	}

	// Find the average of the lagging and leading windows
    float threshold = (sumTrain / (numTrain * 2));

	// If the current cell is above the local average, target detected
    if (input[tid] > threshold * rateFA) 
	{
        detections[tid] = 1; // Detected target
    } 
	else 
	{
        detections[tid] = 0; // No target
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




















/*__global__ void oldcfarKernel(const float* input, int* detections, unsigned int numCells) 
{
	// Define parameters for CFAR
	const int numTrain = 5;   // Number of training cells
	const int numGuard = 1;    // Number of guard cells
	const float rateFA = 0.1; // Desired false alarm rate
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate the CFAR threshold for the current cell
    float sumTrain = 0.0f;
    for (int i = tid - numTrain; i <= tid + numTrain; ++i) 
	{
        if (i >= 0 && i < numCells) 
		{
            sumTrain += input[i];
        }
    }
    float threshold = (sumTrain / (numTrain * 2)) * rateFA;

    // Check if the current cell exceeds the threshold
    if (tid >= numTrain + numGuard*2 && tid < numCells - numTrain - numGuard*2) 
	{
        float sumGuard = 0.0f;
        for (int i = tid - numGuard; i <= tid + numGuard; ++i) 
		{
            if (i != tid) 
			{
                sumGuard += input[i];
            }
        }
		if (tid == 1153) { printf("threshold: %f | sumGuard: %f | sumTrain: %f | input[%d]: %f \n", threshold, sumGuard, sumTrain, tid, input[tid]); }

        if (input[tid] > threshold * sumGuard) 
		{
            detections[tid] = 1; // Detected target
        } 
		else 
		{
            detections[tid] = 0; // No target
        }
    }
} */
