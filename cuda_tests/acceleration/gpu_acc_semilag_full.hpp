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

#include <Eigen/Geometry>
#include <Eigen/Core>

#include "vlasovsolver/cpu_acc_transform.hpp"
#include "vlasovsolver/cpu_acc_intersections.hpp"
#include "vlasovsolver/cpu_acc_map.hpp"

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
void map_column(full_grid_t *full_grid, Real intersection, Real intersection_di, Real intersection_dj, Real intersection_dk, int dimension) {
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

   

  Real *column_data = new Real[column_size + 2*WID]; // propagate needs the extra cells
  Real *target_column_data = new Real[column_size+2*WID];
  for (int block_i = 0; block_i < block_di; block_i++) {
    for (int block_j = 0; block_j < block_dj; block_j++) {
      int blockid = block_i * block_indices_to_id[0] + block_j * block_indices_to_id[1]; // Here k = 0
      for (int cell_i = 0; cell_i < WID; cell_i++) {
        for (int cell_j = 0; cell_j < WID; cell_j++) {
          int cellid = cell_i * cell_indices_to_id[0] + cell_j * cell_indices_to_id[1]; // Here k = 0
          // Construct a temporary array with only data from one column of velocity CELLS
          for (int block_k = 0; block_k < block_dk; block_k++) {
            for (int cell_k = 0; cell_k < WID; ++cell_k) {
              column_data[block_k*WID + cell_k + WID] = full_grid->grid[(blockid+block_k*block_indices_to_id[2])*WID3 + cellid + cell_k*cell_indices_to_id[2]]; // Cells in the same k column in a block are WID2 apart
            }
          }
          if (dimension == 2 && full_grid->min_x + block_i == 15 && full_grid->min_y + block_j == 15 && cell_i == 1 && cell_j == 1) {
            fprint_column("input_column.dat", column_data, column_size, full_grid->min_z);
            printf("%e %e %e %e\n", intersection, intersection_di, intersection_dj, intersection_dk);
          }
          propagate(column_data, target_column_data, block_dk, v_min, cell_dv,
             block_i, cell_i, block_j, cell_j,
             intersection, intersection_di, intersection_dj, intersection_dk);
          //propagate_old(column_data, block_dk, v_min, cell_dv,
          //   block_i, cell_i,block_j, cell_j,
          //   intersection, intersection_di, intersection_dj, intersection_dk);
          if (dimension == 2 && full_grid->min_x + block_i == 15 && full_grid->min_y + block_j == 15 && cell_i == 1 && cell_j == 1)
            //fprint_column("output_column.dat", column_data, column_size, full_grid->min_z);
            fprint_column("output_column.dat", target_column_data, column_size+2*WID, full_grid->min_z);

          // Copy back to full grid
          for (int block_k = 0; block_k < block_dk; block_k++) {
            for (int cell_k = 0; cell_k < WID; ++cell_k) {
              full_grid->grid[(blockid+block_k*block_indices_to_id[2])*WID3 + cellid + cell_k*cell_indices_to_id[2]] = target_column_data[block_k*WID + cell_k + WID];
              //full_grid->grid[(blockid+block_k*block_indices_to_id[2])*WID3 + cellid + cell_k*cell_indices_to_id[2]] = column_data[block_k*WID + cell_k + WID];
            }
          }
        }
      }
    }
  }
  delete[] column_data;
  delete[] target_column_data;
}

void gpu_accelerate_cell_(SpatialCell* spatial_cell,const Real dt) {
   double t1=MPI_Wtime();
   /*compute transform, forward in time and backward in time*/
   phiprof::start("compute-transform");
   //compute the transform performed in this acceleration
   Transform<Real,3,Affine> fwd_transform= compute_acceleration_transformation(spatial_cell,dt);
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
   full_grid_t *full_grid = to_full_grid(spatial_cell);
   //printf("BB: %i %i %i, %i %i %i\n", full_grid->min_x, full_grid->min_y, full_grid->min_z, full_grid->min_x + full_grid->dx, full_grid->min_y + full_grid->dy, full_grid->min_z + full_grid->dz);


   //Do the actual mapping
   map_column(full_grid, intersection_z,intersection_z_di,intersection_z_dj,intersection_z_dk, 2);
   map_column(full_grid, intersection_x,intersection_x_di,intersection_x_dj,intersection_x_dk, 0);
   map_column(full_grid, intersection_y,intersection_y_di,intersection_y_dj,intersection_y_dk, 1);

   // Transfer data back to the SpatialCell
   data_to_SpatialCell(spatial_cell, full_grid);
   printf("rel blocks %i\n", relevant_blocks);
   
   // Remove unnecessary blocks
   std::vector<SpatialCell*> neighbor_ptrs;
   spatial_cell->update_velocity_block_content_lists();
   spatial_cell->adjust_velocity_blocks(neighbor_ptrs,true);
   
   phiprof::stop("compute-mapping");
   double t2=MPI_Wtime();
   spatial_cell->parameters[CellParams::LBWEIGHTCOUNTER] += t2 - t1;
}


   

#endif

