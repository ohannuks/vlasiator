// Host parts of gpu acceleration

#include <algorithm>
#include <math.h>
#include <Eigen/Geometry>
#include <Eigen/Core>

#include "gpu_acc_semilag_full.hpp"
#include "vlasovsolver/cpu_acc_intersections.hpp"
#include "vlasovsolver/cpu_acc_transform.hpp"

using namespace Eigen;
using namespace spatial_cell;

void gpu_accelerate_cell(GPU_velocity_grid grid, const Real dt) {
  double t1=MPI_Wtime();
  const uint block_size = 64u;
  // Use the connected spatial_cell for calculating intersections (on cpu) as usual.
  SpatialCell *spatial_cell = grid.cpu_cell;
  /*compute transform, forward in time and backward in time*/
  phiprof::start("compute-transform");
  //compute the transform performed in this acceleration
  Transform<Real,3,Affine> fwd_transform = compute_acceleration_transformation(spatial_cell, dt);
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

  //Do the actual mapping
  // Calculate the number of cells in the "xy"-plane for whatever orientation the grid is assumed to be in the current mapping direction.
  map_column_kernel_wrapper<2>(grid, intersection_x, intersection_x_di, intersection_x_dj, intersection_x_dk);
  CUDACALL(cudaDeviceSynchronize());
  //map_column_kernel<0><<<>>>(full_grid, intersection_x,intersection_x_di,intersection_x_dj,intersection_x_dk, 0);
  //CUDACALL(cudaDeviceSynchronize());
  //map_column_kernel<1><<<>>>(full_grid, intersection_y,intersection_y_di,intersection_y_dj,intersection_y_dk, 1);
  //CUDACALL(cudaDeviceSynchronize());

  // Transfer data back to the SpatialCell
  spatial_cell = grid.toSpatialCell();
  CUDACALL(cudaDeviceSynchronize());
  // Remove unnecessary blocks
  std::vector<SpatialCell*> neighbor_ptrs;
  spatial_cell->update_velocity_block_content_lists();
  spatial_cell->adjust_velocity_blocks(neighbor_ptrs,true);
  
  phiprof::stop("compute-mapping");
  double t2=MPI_Wtime();
  spatial_cell->parameters[CellParams::LBWEIGHTCOUNTER] += t2 - t1;
}