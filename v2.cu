#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define bSize 1024 // block size definition for easy access

void takeInput(const char *filename, int *array, long n);
void printArray(int *array, long n);

// CUDA kernel function to manage each thread's chunk of elements and perform Ising step
__global__ void isingStep(int *dArrayInput, int *dArrayOutput, long n, int threadElements) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    int buffer;
    long i, j, startingIndex, endingIndex;
   	
   	// find starting and ending indices depending on number of work per thread
    startingIndex = idx * threadElements;
    
    if (idx < n * n / threadElements - 1)
    	// every thread takes a chunk of elements
    	endingIndex = startingIndex + threadElements;
    else if (idx == n * n / threadElements - 1)
    	// last one takes the rest
    	endingIndex = n * n;
    else
    	endingIndex = 0;
    
    // repeat ising step for multiple moments
    for(long k = startingIndex; k < endingIndex; k++) {
    	i = k % n;
    	j = k / n;
    		
    	// add all neighboring values    		
		buffer = dArrayInput[(i - 1 + n) % n + j * n]; 		//left	
		buffer += dArrayInput[(i + 1) % n + j * n]; 		//right			
		buffer += dArrayInput[k];
		buffer += dArrayInput[i + ((j - 1 + n) % n) * n]; 	//up
		buffer += dArrayInput[i + ((j + 1) % n) * n]; 		//down
		
		// keep the sign
		dArrayOutput[k] = buffer / abs(buffer);
    }
    return;
}

int main(int argc, char *argv[]) {
	if (argc != 2) {
		printf("Invalid arguments\n");
		return 1;
	}
	long n = atoi(argv[1]);				// dimension taken as argument
	const char *filename = "input.txt"; // input file name
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
	int *dArray1, *dArray2;
	cudaMalloc((void **)&dArray1, n * n * sizeof(int));
	cudaMalloc((void **)&dArray2, n * n * sizeof(int));
	
	///////////////////
	// INITIAL STATE //
	///////////////////
	
	takeInput(filename, hArray, n);
	//printf("Initial state:\n");
	//printArray(hArray, n);
	//printf("\n\n");
	
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
    int threadElements = 2; // work per thread
    
    while(counter < k) {
		// launch the kernel
    	isingStep<<<gridSize, blockSize>>>(dArray1, dArray2, n, threadElements);
    	counter++;
    	if(counter == k) break;
    	isingStep<<<gridSize, blockSize>>>(dArray2, dArray1, n, threadElements);
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
	elapsedTime = (t2.tv_sec - t1.tv_sec);      // sec
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000000.0;   // us to sec
	
	////////////////
	// DISPLAYING //
	////////////////
	
	//printf("\nResult:\n");
	//printArray(hArray, n);
	printf("Successful temination for: n = %ld k = %d\nTime elapsed: %.4f seconds\n", n, k, elapsedTime);	
	
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
	
	for(long i = 0; i < n * n; i ++)
		if(fscanf(file, "%d", &array[i]) != 1) {
				fprintf(stderr, "Error reading from file");
				exit(EXIT_FAILURE);
		}
	fclose(file);
	return;
}

// auxialiary function to print the grid - mainly for checks
void printArray(int *array, long n) {
	for(long i = 0; i < n; i++) {
		for(int j = 0; j < n; j++)
			printf("%d\t", array[i * n + j]);
		printf("\n");
	}
	return;
}
