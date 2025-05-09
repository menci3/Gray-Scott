#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include <cuda_runtime.h>
#include <cuda.h>

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

__global__ void GrayScottKernel(unsigned char *U, unsigned char *V, 
                                unsigned char *Unew, unsigned char *Vnew,
                                float Du, float Dv, float F, float k, float dt,
                                int grid_size) {
    
    // Define shared memory for the tile
    // Add outer cells (1 extra on each side)
    extern __shared__ unsigned char shared_mem[];
    unsigned char *s_U = shared_mem;
    unsigned char *s_V = &s_U[(blockDim.y + 2) * (blockDim.x + 2)];
    
    // Calculate the global and local thread indices
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;  // Starting row
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;  // Starting column
    
    int tx = threadIdx.x + 1;  // Local x-coordinate with outer offset
    int ty = threadIdx.y + 1;  // Local y-coordinate with outer offset
    
    // Local indices for shared memory with outer cells
    int s_width = blockDim.x + 2;
    int s_idx = ty * s_width + tx;
    
    // Global index
    int g_idx = i * grid_size + j;
    
    // Load data into shared memory
    // Each thread loads its own cell
    if (i < grid_size - 1 && j < grid_size - 1) {
        s_U[s_idx] = U[g_idx];
        s_V[s_idx] = V[g_idx];
        
        // Load cells (top, bottom, left, right)
        // Top
        if (threadIdx.y == 0) {
            int up = (i - 1 + grid_size) % grid_size;
            s_U[(ty-1) * s_width + tx] = U[up * grid_size + j];
            s_V[(ty-1) * s_width + tx] = V[up * grid_size + j];
        }
        
        // Bottom
        if (threadIdx.y == blockDim.y - 1 || i == grid_size - 2) {
            int down = (i + 1) % grid_size;
            s_U[(ty+1) * s_width + tx] = U[down * grid_size + j];
            s_V[(ty+1) * s_width + tx] = V[down * grid_size + j];
        }
        
        // Left
        if (threadIdx.x == 0) {
            int left = (j - 1 + grid_size) % grid_size;
            s_U[ty * s_width + (tx-1)] = U[i * grid_size + left];
            s_V[ty * s_width + (tx-1)] = V[i * grid_size + left];
        }
        
        // Right
        if (threadIdx.x == blockDim.x - 1 || j == grid_size - 2) {
            int right = (j + 1) % grid_size;
            s_U[ty * s_width + (tx+1)] = U[i * grid_size + right];
            s_V[ty * s_width + (tx+1)] = V[i * grid_size + right];
        }
    }
    
    __syncthreads();
    
    // Only process interior cells (exclude borders)
    if (i < grid_size - 1 && j < grid_size - 1) {

        float lapU = s_U[s_idx - 1] +          
                     s_U[s_idx + 1] +          
                     s_U[s_idx - s_width] +   
                     s_U[s_idx + s_width] -    
                     4 * s_U[s_idx];           
                     
        float lapV = s_V[s_idx - 1] +          
                     s_V[s_idx + 1] +          
                     s_V[s_idx - s_width] +   
                     s_V[s_idx + s_width] -    
                     4 * s_V[s_idx];           
        
        float zmnozek = s_U[s_idx] * s_V[s_idx] * s_V[s_idx];
        
        Unew[g_idx] = s_U[s_idx] + dt * (-zmnozek + F * (1 - s_U[s_idx]) + Du * lapU);
        Vnew[g_idx] = s_V[s_idx] + dt * (zmnozek - (F + k) * s_V[s_idx] + Dv * lapV);
    }
}

// Function to run the CUDA version and measure execution time
float GrayScottSolverCUDA(unsigned char *h_U, unsigned char *h_V, 
                         float Du, float Dv, float F, float k, float dt, 
                         int steps, int grid_size, int grid_size_whole,
                         dim3 block_size) {
    unsigned char *d_U, *d_V, *d_Unew, *d_Vnew, *d_temp;
    float elapsed_time = 0.0f;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Timing
    cudaEventRecord(start, 0);
    
    cudaMalloc((void**)&d_U, grid_size_whole);
    cudaMalloc((void**)&d_V, grid_size_whole);
    cudaMalloc((void**)&d_Unew, grid_size_whole);
    cudaMalloc((void**)&d_Vnew, grid_size_whole);
    
    cudaMemcpy(d_U, h_U, grid_size_whole, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, grid_size_whole, cudaMemcpyHostToDevice);
    
    // Also accounting for outer blocks
    dim3 grid_dims((grid_size + block_size.x - 3) / block_size.x, 
                  (grid_size + block_size.y - 3) / block_size.y);
    
    // Shared memory size
    size_t shared_mem_size = 2 * (block_size.x + 2) * (block_size.y + 2) * sizeof(unsigned char);
    
    // Main computation loop
    for (int n = 0; n < steps; n++) {
        
        GrayScottKernel<<<grid_dims, block_size, shared_mem_size>>>(
            d_U, d_V, d_Unew, d_Vnew, Du, Dv, F, k, dt, grid_size);
        
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
            break;
        }
        
        d_temp = d_U;
        d_U = d_Unew;
        d_Unew = d_temp;
        
        d_temp = d_V;
        d_V = d_Vnew;
        d_Vnew = d_temp;
    }
    
    cudaMemcpy(h_U, d_U, grid_size_whole, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_V, d_V, grid_size_whole, cudaMemcpyDeviceToHost);
    
    // Timing
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    cudaFree(d_U);
    cudaFree(d_V);
    cudaFree(d_Unew);
    cudaFree(d_Vnew);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return elapsed_time / 1000.0f;
}

int main(int argc, char *argv[]) {
    // Default parameters
    int grid_size = 0;
    dim3 block_size(16, 16);  // Default block size
    int run_sequential = 0;
    int run_parallel = 0;
    
    // Simulation parameters
    float Du = 0.16f;
    float Dv = 0.08f;
    float F  = 0.060f;
    float k  = 0.062f;
    float dt = 1.0f;
    int steps = 5000;
    
    // Parse command line arguments
    if (argc < 2) {
        printf("USAGE: %s grid_size [-s|-p] [-block x y]\n", argv[0]);
        printf("  grid_size: Size of the simulation grid (NxN)\n");
        printf("  -s: Run sequential version\n");
        printf("  -p: Run parallel CUDA version with benchmarking\n");
        printf("  -block x y: Specify custom block dimensions for CUDA (default: 16x16)\n");
        return 1;
    }
    
    // Parse grid size (first argument)
    grid_size = atoi(argv[1]);
    if (grid_size <= 0) {
        fprintf(stderr, "Invalid grid size: %d\n", grid_size);
        return 1;
    }
    
    // Parse remaining flags
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-s") == 0) {
            run_sequential = 1;
            run_parallel = 0;
        } else if (strcmp(argv[i], "-p") == 0) {
            run_sequential = 0;
            run_parallel = 1;
        } else if (strcmp(argv[i], "-block") == 0 && i+2 < argc) {
            block_size.x = atoi(argv[i+1]);
            block_size.y = atoi(argv[i+2]);
            i += 2;
        }
    }
    
    // Default sequential
    if (!run_sequential && !run_parallel) {
        run_sequential = 1;
    }
    
    int grid_size_whole = grid_size * grid_size;
    printf("Grid size: %d x %d\n", grid_size, grid_size);
    
    // Run sequential version
    if (run_sequential) {
        // Allocate memory
        unsigned char *U = (unsigned char *)malloc(grid_size_whole);
        unsigned char *V = (unsigned char *)malloc(grid_size_whole);
        
        if (!U || !V) {
            fprintf(stderr, "Memory allocation failed.\n");
            return 1;
        }
        
        // Initialize grid
        memset(U, 1, grid_size_whole);
        memset(V, 0, grid_size_whole);
        
        for (int i = grid_size / 2 - 10; i < grid_size / 2 + 10; i++) {
            for (int j = grid_size / 2 - 10; j < grid_size / 2 + 10; j++) {
                int index = i * grid_size + j;
                U[index] = 0.75;
                V[index] = 0.25;
            }
        }
        
        printf("Running sequential version...\n");
        clock_t begin = clock();
        
        GrayScottSolver(U, V, Du, Dv, F, k, dt, steps, grid_size, grid_size_whole);
        
        clock_t end = clock();
        float elapsed_s = ((float)(end - begin) / CLOCKS_PER_SEC);
        printf("Sequential method time: %.3f seconds\n", elapsed_s);
        
        free(U);
        free(V);
    }
    
    // Run parallel version with benchmarking
    if (run_parallel) {
        // Define different grid sizes to benchmark
        int grid_sizes[] = {256, 512, 1024, 2048, 4096};
        int num_grid_sizes = 5;
        
        // Define different block sizes to test
        dim3 block_sizes[] = {
            dim3(8, 8),
            dim3(16, 16),
            dim3(32, 8),
            dim3(32, 16),
            dim3(32, 32)
            
            // TODO: Add more?
        };
        int num_block_sizes = 5;
        
        printf("Gray-Scott CUDA Benchmarking\n");
        printf("Running %d steps for each configuration\n\n", steps);
        
        // Print header
        printf("%-10s %-15s %-15s\n", "Grid Size", "Block Size", "Time (s)");
        printf("----------------------------------------\n");
        
        // Track the best configuration
        float best_time = INFINITY;
        int best_grid_size = 0;
        dim3 best_block_size(0, 0);
        
        // If grid size provided, benchmark only that one
        if (grid_size > 0) {
            num_grid_sizes = 1;
            grid_sizes[0] = grid_size;
        }
        
        // Run benchmarks for each grid size
        for (int i = 0; i < num_grid_sizes; i++) {
            int current_grid_size = grid_sizes[i];
            int current_grid_whole = current_grid_size * current_grid_size;
            
            printf("\nTesting grid size %dx%d\n", current_grid_size, current_grid_size);
            printf("----------------------------------------\n");
            
            // Test different block sizes for CUDA version
            for (int j = 0; j < num_block_sizes; j++) {
                dim3 current_block_size = block_sizes[j];
                
                // Skip invalid configurations (block too large)
                if (current_block_size.x * current_block_size.y > 1024) {
                    printf("%-10d %-3dx%-3d %13s\n", current_grid_size, 
                           current_block_size.x, current_block_size.y, "Too large");
                    continue;
                }
                
                // Allocate memory for CUDA version
                unsigned char *U_cuda = (unsigned char *)malloc(current_grid_whole);
                unsigned char *V_cuda = (unsigned char *)malloc(current_grid_whole);
                
                if (!U_cuda || !V_cuda) {
                    fprintf(stderr, "Memory allocation failed for grid size %d\n", current_grid_size);
                    continue;
                }
                
                // Initialize grid
                memset(U_cuda, 1, current_grid_whole);
                memset(V_cuda, 0, current_grid_whole);
                
                for (int i = current_grid_size / 2 - 10; i < current_grid_size / 2 + 10; i++) {
                    for (int j = current_grid_size / 2 - 10; j < current_grid_size / 2 + 10; j++) {
                        int index = i * current_grid_size + j;
                        U_cuda[index] = 0.75;
                        V_cuda[index] = 0.25;
                    }
                }
                
                float cuda_time = GrayScottSolverCUDA(U_cuda, V_cuda, Du, Dv, F, k, dt, 
                                                    steps, current_grid_size, current_grid_whole, 
                                                    current_block_size);
                
                printf("%-10d %-3dx %-3d %13.3f\n", current_grid_size, 
                       current_block_size.x, current_block_size.y, cuda_time);
                
                if (cuda_time < best_time) {
                    best_time = cuda_time;
                    best_grid_size = current_grid_size;
                    best_block_size = current_block_size;
                }
                
                free(U_cuda);
                free(V_cuda);
            }
        }
        
        printf("\nOptimal configuration:\n");
        printf("Grid size: %dx%d, Block size: %dx%d, Time: %.3f seconds\n", 
               best_grid_size, best_grid_size, best_block_size.x, best_block_size.y, best_time);
    }
    
    return 0;
}