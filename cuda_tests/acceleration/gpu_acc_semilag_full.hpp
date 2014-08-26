/*
  This file is part of Vlasiator.
  Copyright 2013,2014 Finnish Meteorological Institute
*/

#ifndef GPU_ACC_SEMILAG_FULL_H
#define GPU_ACC_SEMILAG_FULL_H

#ifdef ACC_SEMILAG_PLM
#define RECONSTRUCTION_ORDER 1
#endif
#ifdef ACC_SEMILAG_PPM
#define RECONSTRUCTION_ORDER 2
#endif

#include "spatial_cell.hpp"
#include "../gpu_velocity_grid.hpp"

#define index(i,j,k)   ( k + WID + j * (blocks_per_dim_z + 2) * WID + i * (blocks_per_dim_z + 2) * blocks_per_dim_y * WID2 )
#define colindex(i,j)   ( j * (blocks_per_dim_z + 2) * WID + i * (blocks_per_dim_z + 2) * blocks_per_dim_y * WID2 )

using namespace std;

void gpu_accelerate_cell(GPU_velocity_grid grid, const Real dt);

void map_column_kernel_wrapper(GPU_velocity_grid grid, Real intersection, Real intersection_di, Real intersection_dj, Real intersection_dk, uint dim);


#endif

