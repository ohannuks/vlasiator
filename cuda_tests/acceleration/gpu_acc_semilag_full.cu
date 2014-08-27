/*
  This file is part of Vlasiator.
  Copyright 2013,2014 Finnish Meteorological Institute
*/

#include "gpu_acc_semilag_full.hpp"

using namespace std;
using namespace spatial_cell;

/*!
  MC limiter. Give absolute value of slope and its sign separately
*/
__device__ void slope_limiter_(const Real& l,const Real& m, const Real& r, Real& slope_abs, Real& slope_sign) {
  Real sign;
  Real a=r-m;
  Real b=m-l; 
  Real minval=fminf(2.0 * abs(a),2.0 * abs(b));
  minval=fminf(minval, (Real)(0.5 * abs(a+b)));
  
  //check for extrema, set absolute value
  slope_abs = a*b < 0 ? 0.0: minval;
  slope_sign = a + b < 0 ? -1.0 : 1.0;

}

__device__ inline Real slope_limiter_(const Real& l,const Real& m, const Real& r) {
  Real sign;
  Real a=r-m;
  Real b=m-l; 
  Real minval=std::min(2.0 * abs(a), 2.0 * abs(b));
  minval=std::min(minval, (Real)(0.5 * abs(a+b)));
  
  //check for extrema
  Real output = a*b < 0 ? 0.0 : minval;
  //set sign
  return a + b < 0 ? -output : output;
}

/*Compute face values of cell k. Based on explicit h6 estimate*/
__device__ inline void compute_h6_face_values(Real *values, Real &fv_l, Real &fv_r, int k) {

   /*compute left value*/
   fv_l = 1.0/60.0 * (values[k - 3 + WID]  - 8.0 * values[k - 2 + WID]  + 37.0 * values[k - 1 + WID] +
          37.0 * values[k  + WID] - 8.0 * values[k + 1 + WID] + values[k + 2 + WID]); // Same as right face of previous cell right face
   /*set right value*/
   ++k;
   fv_r = 1.0/60.0 * (values[k - 3 + WID]  - 8.0 * values[k - 2 + WID]  + 37.0 * values[k - 1 + WID] +
          37.0 * values[k  + WID] - 8.0 * values[k + 1 + WID] + values[k + 2 + WID]);
}


__device__ inline void filter_extrema(Real *values, Real &fv_l, Real &fv_r, int k) {
   //Coella1984 eq. 1.10, detect extrema and make algorithm constant if it is
   Real extrema_check = ((fv_r - values[k + WID]) * (values[k + WID] - fv_l));
   fv_l = extrema_check < 0 ? values[k + WID]: fv_l;
   fv_r = extrema_check < 0 ? values[k + WID]: fv_r;
}

/*Filter according to Eq. 19 in White et al.*/
__device__ inline void filter_boundedness(Real *values, Real &fv_l, Real &fv_r, int k) {
   /*First Eq. 19 & 20*/
   bool do_fix_bounds =
      (values[k - 1 + WID] - fv_l) * (fv_l - values[k + WID]) < 0 ||
      (values[k + 1 + WID] - fv_r) * (fv_r - values[k + WID]) < 0;
   if(do_fix_bounds) {
      Real slope_abs,slope_sign;
      slope_limiter_(values[k -1 + WID], values[k + WID], values[k + 1 + WID], slope_abs, slope_sign);
      //detect and  fix boundedness, as in WHITE 2008
      fv_l = (values[k -1 + WID] - fv_l) * (fv_l - values[k + WID]) < 0 ?
         values[k + WID] - slope_sign * fminf((Real)0.5 * slope_abs, (Real)abs(fv_l - values[k + WID])) :
         fv_l;
      fv_r = (values[k + 1 + WID] - fv_r) * (fv_r - values[k + WID]) < 0 ?
         values[k + WID] + slope_sign * fminf( (Real)0.5 * slope_abs, (Real)abs(fv_r- values[k + WID])) :
         fv_r;
   }
}


/*!
 Compute PLM coefficients
 f(v) = a[0] + a[1]/2.0*t 
 t=(v-v_{i-0.5})/dv where v_{i-0.5} is the left face of a cell
 The factor 2.0 is in the polynom to ease integration, then integral is a[0]*t + a[1]*t**2
*/
__device__ inline void compute_plm_coeff_explicit_columns(Real *values, Real a[RECONSTRUCTION_ORDER + 1], uint k){ 
   const Real d_cv=slope_limiter_(values[k - 1 + WID], values[k + WID], values[k + 1 + WID]);
   a[0] = values[k + WID] - d_cv * 0.5;
   a[1] = d_cv * 0.5;
}

/*
  Compute parabolic reconstruction with an explicit scheme
  
  Note that value array starts with an empty block, thus values[k + WID]
  corresponds to the current (centered) cell.
*/

__device__ inline void compute_ppm_coeff_explicit_columns(Real *values, Real a[RECONSTRUCTION_ORDER + 1], uint k){
   Real p_face;
   Real m_face;
   Real fv_l; /*left face value, extra space for ease of implementation*/
   Real fv_r; /*right face value*/

   // compute_h6_face_values(values,n_cblocks,fv_l, fv_r); 
   // filter_boundedness(values,n_cblocks,fv_l, fv_r); 
   // filter_extrema(values,n_cblocks,fv_l, fv_r);

   compute_h6_face_values(values,fv_l, fv_r, k);
   filter_boundedness(values,fv_l, fv_r, k);
   filter_extrema(values,fv_l, fv_r, k);
   m_face = fv_l;
   p_face = fv_r;
   
   //Coella et al, check for monotonicity   
   m_face = (p_face - m_face) * (values[k + WID] - 0.5 * (m_face + p_face)) > (p_face - m_face)*(p_face - m_face) / 6.0 ?
      3 * values[k + WID] - 2 * p_face : m_face;
   p_face = -(p_face - m_face) * (p_face - m_face) / 6.0 > (p_face - m_face) * (values[k + WID] - 0.5 * (m_face + p_face)) ?
      3 * values[k + WID] - 2 * m_face : p_face;

   //Fit a second order polynomial for reconstruction see, e.g., White
   //2008 (PQM article) (note additional integration factors built in,
   //contrary to White (2008) eq. 4
   a[0] = m_face;
   a[1] = 3.0 * values[k + WID] - 2.0 * m_face - p_face;
   a[2] = (m_face + p_face - 2.0 * values[k + WID]);
}

// Target needs to be allocated
__device__ void propagate(Real *values, Real *target, uint  blocks_per_dim, Real v_min, Real dv,
       uint i_block, uint i_cell, uint j_block, uint j_cell,
       Real intersection, Real intersection_di, Real intersection_dj, Real intersection_dk){

  Real a[RECONSTRUCTION_ORDER + 1];
  //Real *target = new Real[((int)spatial_cell::SpatialCell::vx_length+2)*WID];
  /*clear target*/
  for (uint k=0; k<WID* (blocks_per_dim + 2); ++k){
       target[k] = 0.0;
  }
   
   /* intersection_min is the intersection z coordinate (z after
      swaps that is) of the lowest possible z plane for each i,j
      index 
   */
  const Real intersection_min = intersection +
     (i_block * WID + i_cell) * intersection_di + 
     (j_block * WID + j_cell) * intersection_dj;
  

  /*compute some initial values, that are used to set up the
   * shifting of values as we go through all blocks in
   * order. See comments where they are shifted for
   * explanations of their meening*/

  /*loop through all blocks in column and compute the mapping as integrals*/
  for (unsigned int k_block = 0; k_block<blocks_per_dim;k_block++) {
    for (uint k_cell=0; k_cell<WID; ++k_cell){ 
      /*v_l, v_r are the left and right velocity coordinates of source cell*/
      Real v_l = v_min + (k_block * WID + k_cell) * dv;
      Real v_r = v_l + dv;
      /*left(l) and right(r) k values (global index) in the target
  lagrangian grid, the intersecting cells. Again old right is new left*/               
      const int target_gk_l = (int)((v_l - intersection_min)/intersection_dk);
      const int target_gk_r = (int)((v_r - intersection_min)/intersection_dk);

      for(int gk = target_gk_l; gk <= target_gk_r; gk++){
         //the velocity limits  for the integration  to put mass
         //in the targe cell. If both v_r and v_l are in same target cell
         //then v_int_l,v_int_r should be between v_l and v_r.
         //v_int_norm_l and v_int_norm_r normalized to be between 0 and 1 in the cell.
  const Real v_int_l = min( max((Real)(gk) * intersection_dk + intersection_min, v_l), v_r);
  const Real v_int_norm_l = (v_int_l - v_l)/dv;
  const Real v_int_r = min((Real)(gk + 1) * intersection_dk + intersection_min, v_r);
  const Real v_int_norm_r = (v_int_r - v_l)/dv;

  uint k = k_block * WID + k_cell;
  #ifdef ACC_SEMILAG_PLM
    compute_plm_coeff_explicit_columns(values, a, k);
  #endif
  #ifdef ACC_SEMILAG_PPM
    compute_ppm_coeff_explicit_columns(values, a, k);
  #endif
   /*compute left and right integrand*/
#ifdef ACC_SEMILAG_PLM
   Real target_density_l =
     v_int_norm_l * a[0] +
     v_int_norm_l * v_int_norm_l * a[1];
   Real target_density_r =
     v_int_norm_r * a[0] +
     v_int_norm_r * v_int_norm_r * a[1];
#endif
#ifdef ACC_SEMILAG_PPM
   Real target_density_l =
     v_int_norm_l * a[0] +
     v_int_norm_l * v_int_norm_l * a[1] +
     v_int_norm_l * v_int_norm_l * v_int_norm_l * a[2];
   Real target_density_r =
     v_int_norm_r * a[0] +
     v_int_norm_r * v_int_norm_r * a[1] +
     v_int_norm_r * v_int_norm_r * v_int_norm_r * a[2];
#endif
   /*total value of integrand, if it is wihtin bounds*/
         if ( gk >= 0 && gk <= blocks_per_dim * WID )
     target[gk + WID] +=  target_density_r - target_density_l;
      }
    }
  }
  /*copy target to values*/
  /*
  for (unsigned int k_block = 0; k_block<blocks_per_dim;k_block++){
     for (uint k=0; k<WID; ++k){
        values[k_block * WID + k + WID] = target[k_block * WID + k + WID];
     }
  }
  */
}

// Analogous to map_1d
template<int dimension> // Using a template should make the switch case hurt performance less.
__global__ void map_column_kernel(GPU_velocity_grid grid, Real intersection, Real intersection_di, Real intersection_dj, Real intersection_dk) {
  Real cell_dv = grid.grid_dims->cell_dv;
  Real v_min;
  Real is_temp;
  int column_size;
  int block_di, block_dj, block_dk, min_i, min_j;
  uint block_indices_to_id[3]; /*< used when computing id of target block */
  uint cell_indices_to_id[3]; /*< used when computing id of target cell in block*/

  // Move the intersection point to correspond to the full grid.
  intersection +=
     (grid.grid_dims->min.x * WID) * intersection_di + 
     (grid.grid_dims->min.y * WID) * intersection_dj +
     (grid.grid_dims->min.z * WID) * intersection_dk;


 switch (dimension){
     case 0:
      /* i and k coordinates have been swapped*/
      /*set cell size in dimension direction*/
      min_i = grid.grid_dims->min.z;
      min_j = grid.grid_dims->min.y;
      v_min = grid.grid_dims->vx_min + grid.grid_dims->min.x * WID * cell_dv;
      column_size = grid.grid_dims->size.x*WID;
      block_di = grid.grid_dims->size.z;
      block_dj = grid.grid_dims->size.y;
      block_dk = grid.grid_dims->size.x;
      /*swap intersection i and k coordinates*/
      is_temp=intersection_di;
      intersection_di=intersection_dk;
      intersection_dk=is_temp;
      /*set values in array that is used to transfer blockindices to id using a dot product*/
      block_indices_to_id[0] = grid.grid_dims->size.x * grid.grid_dims->size.y;
      block_indices_to_id[1] = grid.grid_dims->size.x;
      block_indices_to_id[2] = 1;
      /*set values in array that is used to transfer blockindices to id using a dot product*/
      cell_indices_to_id[0] = WID2;
      cell_indices_to_id[1] = WID;
      cell_indices_to_id[2] = 1;
      break;
    case 1:
      /* j and k coordinates have been swapped*/
      /*set cell size in dimension direction*/
      min_i = grid.grid_dims->min.x;
      min_j = grid.grid_dims->min.z;
      v_min = grid.grid_dims->vy_min + grid.grid_dims->min.y * WID * cell_dv;
      column_size = grid.grid_dims->size.y*WID;
      block_di = grid.grid_dims->size.x;
      block_dj = grid.grid_dims->size.z;
      block_dk = grid.grid_dims->size.y;
      /*swap intersection j and k coordinates*/
      is_temp=intersection_dj;
      intersection_dj=intersection_dk;
      intersection_dk=is_temp;
      /*set values in array that is used to transfer blockindices to id using a dot product*/
      block_indices_to_id[0] = 1;
      block_indices_to_id[1] = grid.grid_dims->size.x * grid.grid_dims->size.y;
      block_indices_to_id[2] = grid.grid_dims->size.x;
      /*set values in array that is used to transfer blockindices to id using a dot product*/
      cell_indices_to_id[0] = 1;
      cell_indices_to_id[1] = WID2;
      cell_indices_to_id[2] = WID;
      break;
    case 2:
      /*set cell size in dimension direction*/
      min_i = grid.grid_dims->min.x;
      min_j = grid.grid_dims->min.y;
      v_min = grid.grid_dims->vz_min + grid.grid_dims->min.z * WID * cell_dv;
      column_size = grid.grid_dims->size.z*WID;
      block_di = grid.grid_dims->size.x;
      block_dj = grid.grid_dims->size.y;
      block_dk = grid.grid_dims->size.z;
      /*set values in array that is used to transfer blockindices to id using a dot product*/
      block_indices_to_id[0] = 1;
      block_indices_to_id[1] = grid.grid_dims->size.x;
      block_indices_to_id[2] = grid.grid_dims->size.x * grid.grid_dims->size.y;
      /*set values in array that is used to transfer blockindices to id using a dot product*/
      cell_indices_to_id[0] = 1;
      cell_indices_to_id[1] = WID;
      cell_indices_to_id[2] = WID2;
     break;
  }

  int block_i = blockIdx.x;
  int block_j = blockIdx.y;
  int cell_i = threadIdx.x;
  int cell_j = threadIdx.y;

  Real *column_data = new Real[column_size + 2*WID]; // propagate needs the extra cells
  Real *target_column_data = new Real[column_size+2*WID];
  int blockid = block_i * block_indices_to_id[0] + block_j * block_indices_to_id[1]; // Here k = 0
  int cellid = cell_i * cell_indices_to_id[0] + cell_j * cell_indices_to_id[1]; // Here k = 0
  // Construct a temporary array with only data from one column of velocity CELLS
  for (int block_k = 0; block_k < block_dk; block_k++) {
    for (int cell_k = 0; cell_k < WID; ++cell_k) {
      column_data[block_k*WID + cell_k + WID] = grid.vel_grid[(blockid+block_k*block_indices_to_id[2])].data[cellid + cell_k*cell_indices_to_id[2]]; // Cells in the same k column in a block are WID2 apart
    }
  }
  propagate(column_data, target_column_data, block_dk, v_min, cell_dv,
     block_i, cell_i, block_j, cell_j,
     intersection, intersection_di, intersection_dj, intersection_dk);
  // Copy back to full grid
  for (int block_k = 0; block_k < block_dk; block_k++) {
    for (int cell_k = 0; cell_k < WID; ++cell_k) {
      grid.vel_grid[(blockid+block_k*block_indices_to_id[2])].data[cellid + cell_k*cell_indices_to_id[2]] = target_column_data[block_k*WID + cell_k + WID];
    }
  }
  
  delete[] column_data;
  delete[] target_column_data;
}

// Instantiate
template __global__ void map_column_kernel<0>(GPU_velocity_grid ggrid, Real intersection, Real intersection_di, Real intersection_dj, Real intersection_dk);
template __global__ void map_column_kernel<1>(GPU_velocity_grid ggrid, Real intersection, Real intersection_di, Real intersection_dj, Real intersection_dk);
template __global__ void map_column_kernel<2>(GPU_velocity_grid ggrid, Real intersection, Real intersection_di, Real intersection_dj, Real intersection_dk);

template<int dimension>
void map_column_kernel_wrapper(GPU_velocity_grid grid, Real intersection, Real intersection_di, Real intersection_dj, Real intersection_dk) {
  dim3 blocks(grid.grid_dims_host->size.x, grid.grid_dims_host->size.y);
  dim3 cells(WID, WID);
  map_column_kernel<dimension><<<blocks, cells>>>(grid, intersection, intersection_di, intersection_dj, intersection_dk);
}

template void map_column_kernel_wrapper<0>(GPU_velocity_grid grid, Real intersection, Real intersection_di, Real intersection_dj, Real intersection_dk);
template void map_column_kernel_wrapper<1>(GPU_velocity_grid grid, Real intersection, Real intersection_di, Real intersection_dj, Real intersection_dk);
template void map_column_kernel_wrapper<2>(GPU_velocity_grid grid, Real intersection, Real intersection_di, Real intersection_dj, Real intersection_dk);

