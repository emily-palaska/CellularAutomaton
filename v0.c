#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

void isingStep(int **gridInput, int **gridOutput, int n);
int **gridInit(int n);
void takeInput(const char *filename, int **array, int n);
void fillGrid(int **grid, int n);
void printGrid(int **grid, int n);

int main(int argc, char *argv[]) {
	if (argc != 2) {
		printf("Invalid arguments\n");
		return 1;
	}
	
	const char *filename = "input.txt"; //input file name
	int n = 10000; 						// dimension
	int k = atoi(argv[1]); 				// number of steps as file argument
	int counter = 0;
	struct timeval t1, t2;				// variables to count elapsed time
	
	// seed the random number generator with the current time
    srand(time(NULL));
	
	// grid initialization
	int **grid1 = gridInit(n);
	int **grid2 = gridInit(n);
	
	if(!grid1 || !grid2)
		return 1;
	
	///////////////////
	// INITIAL STATE //
	///////////////////
		
	//fillGrid(grid1, n); 				//random
	takeInput(filename, grid1, n); 		//from txt file
		
	//printf("\nInitial state:\n");
	//printGrid(grid1, n);
	
	///////////////
	// ALGORITHM //
	///////////////
	
	gettimeofday(&t1, NULL);
	while(counter < k) {
		isingStep(grid1, grid2, n);
		counter++;
		if(counter == k)c break;
		isingStep(grid2, grid1, n);
		counter++;
	}
	gettimeofday(&t2, NULL);
	double elapsedTime;
	elapsedTime = (t2.tv_sec - t1.tv_sec);					// sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000000.0;   // us to s
	
	////////////////
	// DISPLAYING //
	////////////////
	
	//printf("\nResult:\n");
	//if(counter % 2)
		//printGrid(grid2, n);
	//else
		//printGrid(grid1, n);	
	
	printf("Successful temination for: n = %d k = %d\nTime elapsed: %.4f seconds\n", n, k, elapsedTime);
	
	// free allocated memory
	for(int i = 0; i < n; i++) {
		free(grid1[i]);
		free(grid2[i]);
	}
	free(grid1);
	free(grid2);
	
	return 0;
}

// fucntion that perfoms one ising step, ie calculating one moment
// ising step is defined as a 2d transformation where each element takes the sign of the majority
// of its neighboring values' signs
void isingStep(int **gridInput, int **gridOutput, int n) {
	int buffer;
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			// add all neighboring values
			buffer = gridInput[(i - 1 + n) % n][j];
			buffer += gridInput[i][(j - 1 + n) % n];
			buffer += gridInput[i][j];
			buffer += gridInput[(i + 1) % n][j];
			buffer += gridInput[i][(j + 1) % n];
			// keep the sign
			gridOutput[i][j] = buffer / abs(buffer);
		} 
	}
	return;
}

// function that allocates memory for a grid
int **gridInit(int n) {
	// double pointer allocation
	int **grid = (int **)malloc(n * sizeof(int *));
	if (!grid) {
		printf("Memory allocation failed.");
		return NULL;
	}
	
	// single pointer allocation	
	for(int i = 0; i < n; i++) {
		grid[i] = (int *)malloc(n * sizeof(int));
		
		if (!grid[i]) {
			printf("Memory allocation failed.");
			
			// free previously allocated memory before exiting
			for (int j = 0; j < n; j++)
				free(grid[j]);
			
			free(grid);			
			return NULL;		
		}
	}
	
	return grid;
}

// function that takes the initial values of the grid from a txt file
void takeInput(const char *filename, int **array, int n) {
	FILE *file = fopen(filename, "r");
	if(file == NULL) {
		perror("Error opening file");
		exit(EXIT_FAILURE);
	}
	
	for(int i = 0; i < n; i ++)
		for(int j = 0; j < n; j++)
			if(fscanf(file, "%d", &array[i][j]) != 1) {
				fprintf(stderr, "Error reading from file");
				exit(EXIT_FAILURE);
		}
	fclose(file);
	return;
}

// function that takes the initial values of the grid at random
void fillGrid(int **grid, int n) {
	// random value initialization
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++)
			grid[i][j] = ((rand() % 2) * 2) - 1;	
	}

	return;
}

// auxialiary function to print the grid - mainly for checks
void printGrid(int **grid, int n) {
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			if(grid[i][j] >= 0) printf("+");
			printf("%d\t", grid[i][j]);
		}
		printf("\n");
	}
	return;
}




