#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define bSize 1024 // block size for easy access

void takeInput(const char *filename, int *array, long n);
void printArray(int *array, long n);

// CUDA kernel to perform vector ising step
__global__ void isingStep(int *dArrayInput, int *dArrayOutput, long n) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the thread index is within the array bounds
    if (idx < n * n) {
    	int buffer;
		int __shared__ blockMemory[bSize];
    	int neighbors[5];
    	
    	// copy host data to shared memory
    	blockMemory[idx % bSize] = dArrayInput[idx];
    	__syncthreads();
    	
    	// auxiliary indices for checking border values
    	long i = idx % n;
    	long j = idx / n;
    	long up, down, left, right;
    	
    	left = (i - 1 + n) % n + j * n;		
		right = (i + 1) % n + j * n;			
		up = i + ((j - 1 + n) % n) * n;
		down = i + ((j + 1) % n) * n;
    	
    	// when in the same block, shared memory is acessed
    	// else cpu memory is accesed for border values
    	if (left / bSize == idx / bSize)
    		neighbors[0] = blockMemory[left % bSize];
    	else
    		neighbors[0] = dArrayInput[left];
    	
    	if (right / bSize == idx / bSize)
    		neighbors[1] = blockMemory[right % bSize];
    	else
    		neighbors[1] = dArrayInput[right];
    	
    	neighbors[2] = blockMemory[idx % bSize];
    	
    	if (up / bSize == idx / bSize)
    		neighbors[3] = blockMemory[up % bSize];
    	else
    		neighbors[3] = dArrayInput[up];
    		
    	if (down / bSize == idx / bSize)
    		neighbors[4] = blockMemory[down % bSize];
    	else
    		neighbors[4] = dArrayInput[down];
    		
    	__syncthreads();
    	
    	// add all neighboring values
        buffer = 0;
		for(int k = 0; k < 5; k++)
			buffer += neighbors[k];
		
		// keep the sign
		dArrayOutput[idx] = buffer / abs(buffer);
    }
}

int main(int argc, char *argv[]) {
	if (argc != 2) {
		printf("Invalid arguments\n");
		return 1;
	}
	
	const char *filename = "input.txt"; //input file name
	long n = atoi(argv[1]); 			// dimension taken as argument
	int k = 500; 						// number of steps
	int counter = 0;
	struct timeval t1, t2;				// variables for elapsed time
	
	///////////////////////	
	// MEMORY ALLOCATION //
	///////////////////////
	
	// host
	int *hArray = (int *)malloc(n * n * sizeof(int));
	if(!hArray) {
		perror("Memory allocation failed");
		return 1;
	}
			
	// device	
	int *dArray1;
	int *dArray2;	
	cudaMalloc((void **)&dArray1, n * n * sizeof(int));
	cudaMalloc((void **)&dArray2, n * n * sizeof(int));
	
	///////////////////
	// INITIAL STATE //
	///////////////////
	
	takeInput(filename, hArray, n);
	//printf("Initial state:\n");
	//printArray(hArray, n);
	//printf("\n\nFinal state:\n");
	
	///////////////
	// ALGORITHM //
	///////////////
	
	// start timer
	gettimeofday(&t1, NULL);
	
	// copy host vectors to device
    cudaMemcpy(dArray1, hArray, n * n * sizeof(int), cudaMemcpyHostToDevice);
	
	// define the grid and block dimensions
    int blockSize = bSize;
    int gridSize = (n * n + blockSize - 1) / blockSize;
    
    while(counter < k) {
		// Launch the kernel
    	isingStep<<<gridSize, blockSize>>>(dArray1, dArray2, n);
    	counter++;
    	if(counter == k) break;
    	isingStep<<<gridSize, blockSize>>>(dArray2, dArray1, n);
    	counter++;
	}
    // copy the result back to the host
    if(counter % 2)    
    	cudaMemcpy(hArray, dArray2, n * n * sizeof(int), cudaMemcpyDeviceToHost);
    else
    	cudaMemcpy(hArray, dArray1, n * n * sizeof(int), cudaMemcpyDeviceToHost);
	
	// stop timer
	gettimeofday(&t2, NULL);
	double elapsedTime;
	elapsedTime = (t2.tv_sec - t1.tv_sec);      			// sec
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000000.0;   // us to sec
	
	////////////////
	// DISPLAYING //
	////////////////
	
	//printArray(hArray, n);
	printf("Successful temination for: n = %d k = %d\nTime elapsed: %.4f seconds\n", n, k, elapsedTime);
	
	//////////////////////////
	// MEMORY DE-ALLOCATION //
	//////////////////////////
	
	free(hArray);		// host
    cudaFree(dArray1);	// device
    cudaFree(dArray2);
	return 0;
}

// function that takes the initial values of the grid from a txt file
void takeInput(const char *filename, int *array, long n) {
	FILE *file = fopen(filename, "r");
	if(file == NULL) {
		perror("Error opening file");
		exit(EXIT_FAILURE);
	}
	
	for(int i = 0; i < n * n; i ++)
		if(fscanf(file, "%d", &array[i]) != 1) {
				fprintf(stderr, "Error reading from file");
				exit(EXIT_FAILURE);
		}
	fclose(file);
	return;
}

// auxialiary function to print the grid - mainly for checks
void printArray(int *array, long n) {
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
				if(array[i * n + j] >= 0) printf("+");
				printf("%d ", array[i * n + j]);
			}
		printf("\n");
	}
	return;
}

