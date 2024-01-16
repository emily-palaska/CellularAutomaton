#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int isingStep(int **gridInput, int **gridOutput, int n);
int **gridInit(int n);
void fillGrid(int **grid, int n);
void printGrid(int **grid, int n);

int main() {
	// seed the random number generator with the current time
    srand(time(NULL));

	// define problem dimension
	int n = 5;
	
	// grid initialization
	int **grid1 = gridInit(n);
	int **grid2 = gridInit(n);
	
	if(!grid1 || !grid2)
		return 1;
	
	// initial state randomization
	fillGrid(grid1, n);
	
	// print grids
	printf("\nGrid 1:\n");
	printGrid(grid1, n);
	printf("\nGrid 2:\n");
	printGrid(grid2, n);
	
	int counter;
	while(counter < 100) {
		if(isingStep(grid1, grid2, n) == 0)
			break;
		counter++;
		if(isingStep(grid2, grid1, n) == 0)
			break;
		counter++;
	}
	printf("\n\nCounter: %d\n\n", counter);
	
	// print grids
	printf("\n---RESULTS---\nGrid 1:\n");
	printGrid(grid1, n);
	printf("\nGrid 2:\n");
	printGrid(grid2, n);	
	
	// free allocated memory
	for(int i = 0; i < n; i++) {
		free(grid1[i]);
		free(grid2[i]);
	}
	free(grid1);
	free(grid2);
	
	return 0;
}

// returns 0 if the output doesn't change
int isingStep(int **gridInput, int **gridOutput, int n) {
	int buffer;
	int flag = 0;
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
			if(gridOutput[i][j] != gridInput[i][j]) flag = 1;
		} 
	}
	return flag;
}

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

void fillGrid(int **grid, int n){
	// random value initialization
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++)
			grid[i][j] = ((rand() % 2) * 2) - 1;	
	}

	return;
}

void printGrid(int **grid, int n) {
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++)
			printf("%d\t", grid[i][j]);
		printf("\n");
	}
	return;
}




