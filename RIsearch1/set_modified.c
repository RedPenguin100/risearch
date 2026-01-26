#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#include "set_modified.h"

char buffer[32000];
extern short modified_dsm[6][6][6][6];
extern short modified_dsm_extend[6][6][6][6];

void set_modified_dsm(short *p_dsm, const char *csv_filename) {
    FILE *file;
    short number;
    file = fopen(csv_filename, "r");
    if (file == NULL) {
        printf("Could not open the file.\n");
    }
    int i = 0;
    while (fscanf(file, "%hd,", &number) != EOF) {
        *(p_dsm + i) = number;
        i++;
    }
    fclose(file);
}
