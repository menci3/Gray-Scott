#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

//#include <cuda_runtime.h>
//#include <cuda.h>

//#define STB_IMAGE_IMPLEMENTATION
//#define STB_IMAGE_WRITE_IMPLEMENTATION
//#include "stb_image.h"
//#include "stb_image_write.h"

void GrayScottSolver(unsigned char *U, unsigned char *V , float Du, float Dv, float F,  float k, float dt, int steps, int grid_size, int grid_size_whole) {
    unsigned char *Unew = (unsigned char *)malloc(grid_size_whole);
    unsigned char *Vnew = (unsigned char *)malloc(grid_size_whole);

    float lapU, lapV, zmnozek;


    for (int n = 1; n <= steps; n++) {
        for (int i = 1; i < grid_size - 1; i++) {
            for (int j = 1; j < grid_size - 1; j++) {
                int up    = (i - 1 + grid_size) % grid_size;
                int down  = (i + 1) % grid_size;
                int left  = (j - 1 + grid_size) % grid_size;
                int right = (j + 1) % grid_size;

                int index       = i * grid_size + j;
                int index_up    = up * grid_size + j;
                int index_down  = down * grid_size + j;
                int index_left  = i * grid_size + left;
                int index_right = i * grid_size + right;

                lapU = U[index_left] + U[index_right] + U[index_up] + U[index_down] - 4 * U[index];
                lapV = V[index_left] + V[index_right] + V[index_up] + V[index_down] - 4 * V[index];

                zmnozek = U[index] * V[index] * V[index];

                Unew[index] = U[index] + dt * (-zmnozek + F * (1 - U[index]) + Du * lapU);
                Vnew[index] = V[index] + dt * (zmnozek - (F + k) * V[index] + Dv * lapV);
            }
        }

        // memcpy(U, U_new, grid_size_whole);
        // memcpy(V, V_new, grid_size_whole);

        for (int i = 1; i < grid_size - 1; i++) {
            for (int j = 1; j < grid_size - 1; j++) {
                int index = i * grid_size + j;

                U[index] = Unew[index];
                V[index] = Vnew[index];
            }
        }
    }

    free(Unew);
    free(Vnew);
}


int main(int argc, char *argv[]) {
    // Initialization
    if (argc < 2) {
        printf("USAGE: file_name grid_size\n");
        exit(EXIT_FAILURE);
    }

    printf("Grid size: %d\n", atoi(argv[1]));

    int grid_size = atoi(argv[1]);
    int grid_size_whole = grid_size * grid_size;

    float Du = 0.16f;
    float Dv = 0.08f;
    float F  = 0.060f;
    float k  = 0.062f;
    float dt = 1.0f;
    int steps = 5000;

    unsigned char *U = (unsigned char *)malloc(grid_size_whole);
    unsigned char *V = (unsigned char *)malloc(grid_size_whole);
    
    if (!U || !V) {
        fprintf(stderr, "Memory allocation failed.\n");
        return 1;
    }

    memset(U, 1, grid_size_whole);
    memset(V, 0, grid_size_whole);

    for (int i = grid_size / 2 - 10; i < grid_size / 2 + 10; i++) {
        for (int j = grid_size / 2 - 10; j < grid_size / 2 + 10; j++) {
            int index = i * grid_size + j;
            U[index] = 0.75;
            V[index] = 0.25;
        }
    }

    // Execution
    clock_t begin, end;
    float elapsed_s;

    begin = clock();

    GrayScottSolver(U, V, Du, Dv, F, k, dt, steps, grid_size, grid_size_whole);

    end = clock();
    elapsed_s = ((float)(end - begin) / CLOCKS_PER_SEC);

    printf("Sequential method time: %.3f seconds\n", elapsed_s);

    // Cleanup
    free(U);
    free(V);

    return 0;
}
