#ifndef GPU_VELOCITY_GRID_H
#define GPU_VELOCITY_GRID_H

#include <stdlib.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#include "../spatial_cell.hpp"
#include "spatial_cell_funcs.hpp"

#define ERROR_CELL -1.0f
#define ERROR_BLOCK NULL

#define block_print_format "%5i(%03u,%03u,%03u)%+5.2e, "

#define CUDACALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=false)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPU error(%i): %s %s %d\n", code, cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Returns the the next multiple of divisor of the equivalent float division. Intended to be used with integer types as this assumes integer arithmetic.
template<typename T>
inline T ceilDivide(T dividend, T divisor) {
    return (dividend + divisor - 1) / divisor;
}

// 3d indices
struct ind3d{unsigned int x,y,z;};
// analogous to class VelocityBlock of SpatialCell
typedef struct{Real data[WID3];} vel_block;

// Struct containing the minimum indices of the full grid and size of each dimension in blocks.
typedef struct{
    ind3d min;
    ind3d max;
    ind3d size;
    ind3d sparse_size; // vx_length etc. from SpatialCell
    Real cell_dv;
    Real vx_min, vy_min, vz_min;
    } grid_dims_t;

class GPU_velocity_grid {
    // Pointers point to GPU memory except for the ones labeled host
    public:
        spatial_cell::SpatialCell *cpu_cell; // The SpatialCell on CPU used as input
        Real *block_data; // Blocks from SpatialCell
        unsigned int *velocity_block_list; // Block indices from SpatialCell
        unsigned int *num_blocks, num_blocks_host; // Number of blocks on the SpatialCell. Needed for initializing the full grid on gpu.
        Real *min_val; // From SpatialCell as well
        grid_dims_t *grid_dims, *grid_dims_host; // Contains the metadata of the full grid.
        vel_block *vel_grid; // Block structured full grid

        
        GPU_velocity_grid(spatial_cell::SpatialCell *spacell);
        __host__ __device__ ~GPU_velocity_grid(void); // Dummy destructor that does not do anything to make passing by value possible.
        __host__ void del(void); // The actual destructor used to free memory

        __host__ void init_grid(void);
        __host__ void accelerate(Real);
        __host__ spatial_cell::SpatialCell *toSpatialCell(void);
        __host__ unsigned int min_ind(void);
        __host__ unsigned int max_ind(void);

        // Accessor functions. The blockid here refers to the blockid in the sparse grid.
        __device__ vel_block* get_velocity_grid_block(unsigned int blockid);
        __device__ int full_to_sparse_ind(unsigned int blockid);
        __host__   int full_to_sparse_ind_host(unsigned int blockid);
        __host__   ind3d get_full_grid_block_indices_host(const unsigned int blockid);
        __device__ Real get_velocity_cell(unsigned int blockid, unsigned int cellid);
        __device__ Real set_velocity_cell(unsigned int blockid, unsigned int cellid, Real val);
        __device__ void set_velocity_block(unsigned int blockid, Real *vals);
        
        // Printing and helper functions
        __device__ ind3d get_velocity_block_indices(const unsigned int blockid);
        __device__ ind3d get_full_grid_block_indices(const unsigned int blockid);
        __host__   static ind3d get_velocity_block_indices_host(const unsigned int blockid);
        __device__ unsigned int get_velocity_block(const ind3d indices);
        __host__   void print_velocity_block_list(void);
};

#endif
