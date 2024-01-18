#include <stdio.h>
#include <stdlib.h>

void takeInput(const char *filename, int *array, int n);
void printArray(int *array, int n);

// CUDA kernel to perform vector ising step
__global__ void isingStep(int *dArrayInput, int *dArrayOutput, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int buffer;
    // Check if the thread index is within the array bounds
    
    if (idx < n * n) {
    	int i = idx % n;
    	int j = idx / n;
    	
        // add all neighboring values
		buffer = dArrayInput[(i - 1 + n) % n + j * n];		//left
		buffer += dArrayInput[(i + 1) % n + j * n];			//right
		buffer += dArrayInput[idx];
		buffer += dArrayInput[i + ((j - 1 + n) % n) * n];	//up
		buffer += dArrayInput[i + ((j + 1) % n) * n];		//down
		
		// keep the sign
		dArrayOutput[idx] = buffer / abs(buffer);
    }
}

int main() {
	int n = 7; // dimension
	const char *filename = "input.txt"; //input file name
	int k = 10; // number of steps
	int counter = 0;
	
	// MEMORY ALLOCATION
	
	// host array initialization
	int *hArray = (int *)malloc(n * n * sizeof(int));
	
	if(!hArray) {
		perror("Memory allocation failed");
		return 1;
	}	
		
	// device array initialization	
	int *dArray1;
	int *dArray2;
	
	cudaMalloc((void **)&dArray1, n * n * sizeof(int));
	cudaMalloc((void **)&dArray2, n * n * sizeof(int));
	
	// INITIAL STATE
	takeInput(filename, hArray, n);
	printf("Initial state:\n");
	printArray(hArray, n);
	
	// ALGORITHM APPLICATION
	
	// copy host vectors to device
    cudaMemcpy(dArray1, hArray, n * n * sizeof(int), cudaMemcpyHostToDevice);
	
	// Define the grid and block dimensions
    int blockSize = 256;
    int gridSize = (n * n + blockSize - 1) / blockSize;
    
    while(counter < k) {
		// Launch the kernel
    	isingStep<<<gridSize, blockSize>>>(dArray1, dArray2, n);
    	counter++;
    	if(counter == k) break;
    	isingStep<<<gridSize, blockSize>>>(dArray2, dArray1, n);
    	counter++;
	}
    // Copy the result back to the host
    if(counter % 2)    
    	cudaMemcpy(hArray, dArray1, n * n * sizeof(int), cudaMemcpyDeviceToHost);
    else
    	cudaMemcpy(hArray, dArray2, n * n * sizeof(int), cudaMemcpyDeviceToHost);
	
	// RESULT PRINTING
	printf("\nResult:\n");
	printArray(hArray, n);
	
	
	// MEMORY DE-ALLOCATION
	
	// Host
	free(hArray);

	// Device
    cudaFree(dArray1);
    cudaFree(dArray2);
	return 0;
}

void takeInput(const char *filename, int *array, int n) {
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

void printArray(int *array, int n) {
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++)
			printf("%d\t", array[i * n + j]);
		printf("\n");
	}
	return;
}
