/*
  This file is part of Vlasiator.
  Copyright 2013,2014 Finnish Meteorological Institute
*/

#ifndef CPU_ACC_SEMILAG_FULL_H
#define CPU_ACC_SEMILAG__FULL_H

#ifdef ACC_SEMILAG_PLM
#define RECONSTRUCTION_ORDER 1
#endif
#ifdef ACC_SEMILAG_PPM
#define RECONSTRUCTION_ORDER 2
#endif

#include "algorithm"
#include "cmath"
#include "utility"

/*TODO - replace with standard library c++11 functions as soon as possible*/
#include "boost/array.hpp"
#include "boost/unordered_map.hpp"

#include "common.h"
#include "../../spatial_cell.hpp"
#include "gpu_velocity_grid.hpp"

#include <Eigen/Geometry>
#include <Eigen/Core>

#include "vlasovsolver/cpu_acc_transform.hpp"
#include "vlasovsolver/cpu_acc_intersections.hpp"

#include "map_3d.hpp"

using namespace std;
using namespace spatial_cell;

// Outputs the elements of the given array with the given size to a file
void fprint_column(const char *filename, Real *column, const uint size, const uint min_ind) {
  FILE *filep = fopen(filename, "w");
  for (int i = 0; i < size; i+=WID) {
    fprintf(filep, "%2u ", min_ind + i/WID);
    fprintf(filep, "%3.2e %3.2e %3.2e %3.2e\n", column[i+WID], column[i+1+WID], column[i+2+WID], column[i+3+WID]);
  }
}

// Analogous to map_1d
template<int dimension> // Using a template could make the switch case hurt performance less. At least it should not make anything worse.
__global__ void map_column_kernel(GPU_velocity_grid ggrid, Real intersection, Real intersection_di, Real intersection_dj, Real intersection_dk) {
  Real cell_dv, v_min;
  Real is_temp;
  int column_size;
  int block_di, block_dj, block_dk, min_i, min_j;
  uint max_v_length;
  uint block_indices_to_id[3]; /*< used when computing id of target block */
  uint cell_indices_to_id[3]; /*< used when computing id of target cell in block*/

  // Move the intersection point to correspond to the full grid.
  intersection +=
     (full_grid->min_x * WID) * intersection_di + 
     (full_grid->min_y * WID) * intersection_dj +
     (full_grid->min_z * WID) * intersection_dk;


 switch (dimension){
     case 0:
      /* i and k coordinates have been swapped*/
      /*set cell size in dimension direction*/
      min_i = full_grid->min_z;
      min_j = full_grid->min_y;
      cell_dv=SpatialCell::cell_dvx; 
      v_min=SpatialCell::vx_min + full_grid->min_x * WID * cell_dv;
      column_size = full_grid->dx*WID;
      block_di = full_grid->dz;
      block_dj = full_grid->dy;
      block_dk = full_grid->dx;
      /*swap intersection i and k coordinates*/
      is_temp=intersection_di;
      intersection_di=intersection_dk;
      intersection_dk=is_temp;
      /*set values in array that is used to transfer blockindices to id using a dot product*/
      block_indices_to_id[0]=full_grid->dx * full_grid->dy;
      block_indices_to_id[1]=full_grid->dx;
      block_indices_to_id[2]=1;
      /*set values in array that is used to transfer blockindices to id using a dot product*/
      cell_indices_to_id[0]=WID2;
      cell_indices_to_id[1]=WID;
      cell_indices_to_id[2]=1;
      break;
    case 1:
      /* j and k coordinates have been swapped*/
      /*set cell size in dimension direction*/
      min_i = full_grid->min_x;
      min_j = full_grid->min_z;
      cell_dv=SpatialCell::cell_dvy;
      v_min=SpatialCell::vy_min + full_grid->min_y * WID * cell_dv;
      column_size = full_grid->dy*WID;
      block_di = full_grid->dx;
      block_dj = full_grid->dz;
      block_dk = full_grid->dy;
      /*swap intersection j and k coordinates*/
      is_temp=intersection_dj;
      intersection_dj=intersection_dk;
      intersection_dk=is_temp;
      /*set values in array that is used to transfer blockindices to id using a dot product*/
      block_indices_to_id[0]=1;
      block_indices_to_id[1]=full_grid->dx * full_grid->dy;
      block_indices_to_id[2]=full_grid->dx;
      /*set values in array that is used to transfer blockindices to id using a dot product*/
      cell_indices_to_id[0]=1;
      cell_indices_to_id[1]=WID2;
      cell_indices_to_id[2]=WID;
      break;
    case 2:
      /*set cell size in dimension direction*/
      min_i = full_grid->min_x;
      min_j = full_grid->min_y;
      cell_dv=SpatialCell::cell_dvz;
      v_min=SpatialCell::vz_min + full_grid->min_z * WID * cell_dv;
      column_size = full_grid->dz*WID;
      block_di = full_grid->dx;
      block_dj = full_grid->dy;
      block_dk = full_grid->dz;
      /*set values in array that is used to transfer blockindices to id using a dot product*/
      block_indices_to_id[0]=1;
      block_indices_to_id[1]=full_grid->dx;
      block_indices_to_id[2]=full_grid->dx * full_grid->dy;
      /*set values in array that is used to transfer blockindices to id using a dot product*/
      cell_indices_to_id[0]=1;
      cell_indices_to_id[1]=WID;
      cell_indices_to_id[2]=WID2;
     break;
  }

  uint i = blockIdx.x*blockDim.x + threadIdx.x;

  Real *column_data = new Real[column_size + 2*WID]; // propagate needs the extra cells
  Real *target_column_data = new Real[column_size+2*WID];
  int blockid = block_i * block_indices_to_id[0] + block_j * block_indices_to_id[1]; // Here k = 0
  int cellid = cell_i * cell_indices_to_id[0] + cell_j * cell_indices_to_id[1]; // Here k = 0
  // Construct a temporary array with only data from one column of velocity CELLS
  for (int block_k = 0; block_k < block_dk; block_k++) {
    for (int cell_k = 0; cell_k < WID; ++cell_k) {
      column_data[block_k*WID + cell_k + WID] = full_grid->grid[(blockid+block_k*block_indices_to_id[2])*WID3 + cellid + cell_k*cell_indices_to_id[2]]; // Cells in the same k column in a block are WID2 apart
    }
  }
  propagate(column_data, target_column_data, block_dk, v_min, cell_dv,
     block_i, cell_i, block_j, cell_j,
     intersection, intersection_di, intersection_dj, intersection_dk);
  // Copy back to full grid
  for (int block_k = 0; block_k < block_dk; block_k++) {
    for (int cell_k = 0; cell_k < WID; ++cell_k) {
      full_grid->grid[(blockid+block_k*block_indices_to_id[2])*WID3 + cellid + cell_k*cell_indices_to_id[2]] = target_column_data[block_k*WID + cell_k + WID];
      //full_grid->grid[(blockid+block_k*block_indices_to_id[2])*WID3 + cellid + cell_k*cell_indices_to_id[2]] = column_data[block_k*WID + cell_k + WID];
    }
  }
  
  delete[] column_data;
  delete[] target_column_data;
}

// Instantiate
template void map_column_kernel<0>();
template void map_column_kernel<1>();
template void map_column_kernel<2>();

void gpu_accelerate_cell(GPU_velocity_grid grid, const Real dt) {
  double t1=MPI_Wtime();
  const uint block_size = 64u;
  // Use the connected spatial_cell for calculating intersections (on cpu) as usual.
  SpatialCell *spatial_cell = grid.cpu_cell;
  /*compute transform, forward in time and backward in time*/
  phiprof::start("compute-transform");
  //compute the transform performed in this acceleration
  Transform<Real,3,Affine> fwd_transform= compute_acceleration_transformation(spatial_cell, dt);
  Transform<Real,3,Affine> bwd_transform = fwd_transform.inverse();
  phiprof::stop("compute-transform");
  phiprof::start("compute-intersections");
  Real intersection_z,intersection_z_di,intersection_z_dj,intersection_z_dk;
  Real intersection_x,intersection_x_di,intersection_x_dj,intersection_x_dk;
  Real intersection_y,intersection_y_di,intersection_y_dj,intersection_y_dk;
  compute_intersections_z(spatial_cell, bwd_transform, fwd_transform,
                          intersection_z,intersection_z_di,intersection_z_dj,intersection_z_dk);
  compute_intersections_x(spatial_cell, bwd_transform, fwd_transform,
                          intersection_x,intersection_x_di,intersection_x_dj,intersection_x_dk);
  compute_intersections_y(spatial_cell, bwd_transform, fwd_transform,
                          intersection_y,intersection_y_di,intersection_y_dj,intersection_y_dk);
  phiprof::stop("compute-intersections");
  phiprof::start("compute-mapping");

  // Create a full grid from the sparse spatialCell
  GPU_velocity_grid *ggrid = new GPU_velocity_grid(spatial_ cell);


  //Do the actual mapping
  // Calculate the number of cells in the "xy"-plane for whatever orientation the grid is assumed to be in the current mapping direction.
  uint num_cells_ij = grid.grid_dims_host->size.x * grid.grid_dims_host->size.y * WID2;
  uint num_blocks = ceilDivide(num_cells_ij, block_size); // Number of thread blocks
  map_column_kernel<2><<<num_blocks, block_size>>>(full_grid, intersection_z,intersection_z_di,intersection_z_dj,intersection_z_dk);
  //map_column_kernel<0><<<>>>(full_grid, intersection_x,intersection_x_di,intersection_x_dj,intersection_x_dk, 0);
  //map_column_kernel<1><<<>>>(full_grid, intersection_y,intersection_y_di,intersection_y_dj,intersection_y_dk, 1);

  // Transfer data back to the SpatialCell
  spatial_cell = ggrid->data_to_SpatialCell(spatial_cell, full_grid);
  CUDACALL(cudaDeviceSynchronize());
  // Remove unnecessary blocks
  std::vector<SpatialCell*> neighbor_ptrs;
  spatial_cell->update_velocity_block_content_lists();
  spatial_cell->adjust_velocity_blocks(neighbor_ptrs,true);
  
  phiprof::stop("compute-mapping");
  double t2=MPI_Wtime();
  spatial_cell->parameters[CellParams::LBWEIGHTCOUNTER] += t2 - t1;
}


   

#endif

