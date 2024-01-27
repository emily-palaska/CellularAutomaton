#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void generateRandomValues(int n) {
    // Open the file for writing (and clearing existing content)
    FILE *file = fopen("input.txt", "w");

    // Check if the file was opened successfully
    if (file == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    // Seed the random number generator with the current time
    srand((unsigned int)time(NULL));

    // Generate and write random values to the file
    for (int i = 0; i < n * n; ++i) {
        // Generate a random value (either 1 or -1)
        int randomValue = rand() % 2 == 0 ? 1 : -1;

        // Write the random value to the file
        fprintf(file, "%d\n", randomValue);
    }

    // Close the file
    fclose(file);
}

int main(int argc, char *argv[]) {
    int n = pow(2, atoi(argv[1]));
    
    // Call the function to generate random values and clear existing content
    generateRandomValues(n);

    return 0;
}

