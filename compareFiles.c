#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int compareFiles(const char *file1, const char *file2) {
    FILE *fptr1, *fptr2;
    char line1[256], line2[256];
    int lineNum = 1;
	int flag = 0;
    // Open the files
    fptr1 = fopen(file1, "r");
    fptr2 = fopen(file2, "r");

    // Check if files are successfully opened
    if (fptr1 == NULL || fptr2 == NULL) {
        perror("Error opening files");
        exit(EXIT_FAILURE);
    }

    // Read and compare lines
    while (fgets(line1, sizeof(line1), fptr1) != NULL && fgets(line2, sizeof(line2), fptr2) != NULL) {
        if (strcmp(line1, line2) != 0) {
            printf("Difference found at line %d:\n", lineNum);
            printf("%s: %s", file1, line1);
            printf("%s: %s", file2, line2);
            flag = 1;
        }
        lineNum++;
    }

    // Check for extra lines in either file
    while (fgets(line1, sizeof(line1), fptr1) != NULL) {
        printf("Extra line in %s at line %d: %s", file1, lineNum, line1);
        flag = 1;
        lineNum++;
    }

    while (fgets(line2, sizeof(line2), fptr2) != NULL) {
        printf("Extra line in %s at line %d: %s", file2, lineNum, line2);
        lineNum++;
        flag = 1;
    }
	
    // Close the files
    fclose(fptr1);
    fclose(fptr2);
    return flag;
}

int main() {
	char *file1 = "file1.txt";
	char *file2 = "file2.txt";
    if (!compareFiles(file1, file2))
    	printf("Files are exactly the same\n");

    return 0;
}

