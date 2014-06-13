#ifndef GPU_VELOCITY_GRID_H
#define GPU_VELOCITY_GRID_H

#include <stdlib.h>
#include <fstream>
#include <iostream>
#include "../spatial_cell.hpp"

// Start CUDA only part
#ifndef NO_CUDA
#include <cuda_runtime.h>

#define CUDACALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=false)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPU error(%i): %s %s %d\n", code, cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

typedef struct{unsigned int x,y,z;} ind3d;

class GPU_velocity_grid {
    public:
        unsigned int *num_blocks;
        unsigned int *velocity_block_list;        
        float *block_data;
        
        // Functions
		GPU_velocity_grid(spatial_cell::SpatialCell *spacell);
		~GPU_velocity_grid(void);
		__device__ static inline int vx_len(void);
		__device__ static inline int vy_len(void);
		__device__ static inline int vz_len(void);
		__host__ void print_blocks(void);
		__device__ static ind3d get_velocity_block_indices(const unsigned int blockid);
		__host__   static ind3d get_velocity_block_indices_host(const unsigned int blockid);
		__device__ static unsigned int get_velocity_block(const ind3d indices);
		__host__ void k_print_blocks(void);
		__host__ unsigned int min_ind(unsigned int len);
		__host__ unsigned int max_ind(unsigned int len);
};

#endif
// End CUDA only part

void print_constants(void);
void init_spatial_cell_static(void);
void print_blocks(spatial_cell::SpatialCell *cell);
spatial_cell::SpatialCell *create_index_test_cell(void);
spatial_cell::SpatialCell *create_maxwellian(float T, float rho);
void fprint_projection(float *projection, std::string filename);
float *xy_projection(spatial_cell::SpatialCell *spacell);
#endif
