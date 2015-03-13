/* This file is part of Vlasiator.
 * Copyright 2015 Finnish Meteorological Institute.
 * 
 * File:   poisson_solver_sor.cpp
 * Author: sandroos
 *
 * Created on January 15, 2015, 12:45 PM
 */

#include <cstdlib>
#include <iostream>
#include <omp.h>

#include "../logger.h"
#include "../grid.h"

#include "poisson_solver_sor.h"

#ifndef NDEBUG
   #define DEBUG_POISSON_SOR
#endif

using namespace std;

extern Logger logFile;

namespace poisson {

   static const int RED   = 0;
   static const int BLACK = 1;

   vector<CellCache3D> innerCellPointersRED;
   vector<CellCache3D> bndryCellPointersRED;
   vector<CellCache3D> innerCellPointersBLACK;
   vector<CellCache3D> bndryCellPointersBLACK;

   PoissonSolver* makeSOR() {
      return new PoissonSolverSOR();
   }

   PoissonSolverSOR::PoissonSolverSOR(): PoissonSolver() { }

   PoissonSolverSOR::~PoissonSolverSOR() { }

   bool PoissonSolverSOR::initialize() {
      bool success = true;
      bndryCellParams[CellParams::PHI] = 0;
      bndryCellParams[CellParams::PHI_TMP] = 0;
      return success;
   }

   bool PoissonSolverSOR::finalize() {
      bool success = true;
      return success;
   }
   
   bool PoissonSolverSOR::calculateElectrostaticField(dccrg::Dccrg<spatial_cell::SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid) {
      bool success = true;
      SpatialCell::set_mpi_transfer_type(Transfer::CELL_PHI,false);
      
      mpiGrid.start_remote_neighbor_copy_updates(POISSON_NEIGHBORHOOD_ID);
      
      // Calculate electric field on inner cells
      if (Poisson::is2D == true) {
         if (calculateElectrostaticField2D(innerCellPointersRED) == false) success = false;
         if (calculateElectrostaticField2D(innerCellPointersBLACK) == false) success = false;
      } else {
         if (calculateElectrostaticField3D(innerCellPointersRED) == false) success = false;
         if (calculateElectrostaticField3D(innerCellPointersBLACK) == false) success = false;
      }
      
      mpiGrid.wait_remote_neighbor_copy_updates(POISSON_NEIGHBORHOOD_ID);
      
      // Calculate electric field on boundary cells
      if (Poisson::is2D == true) {
         if (calculateElectrostaticField2D(bndryCellPointersRED) == false) success = false;
         if (calculateElectrostaticField2D(bndryCellPointersBLACK) == false) success = false;
      } else {
         if (calculateElectrostaticField3D(bndryCellPointersRED) == false) success = false;
         if (calculateElectrostaticField3D(bndryCellPointersBLACK) == false) success = false;
      }
      
      return success;
   }

   void PoissonSolverSOR::evaluate2D(std::vector<poisson::CellCache3D>& cellPointers,const int& cellColor) {
      const Real weight = 1.5;
      
      #pragma omp for
      for (size_t c=0; c<cellPointers.size(); ++c) {
         #ifdef DEBUG_POISSON_SOR
         bool ok = true;
         if (cellPointers[c][0] == NULL) ok = false;
         if (cellPointers[c][1] == NULL) ok = false;
         if (cellPointers[c][2] == NULL) ok = false;
         if (cellPointers[c][3] == NULL) ok = false;
         if (cellPointers[c][4] == NULL) ok = false;
         if (ok == false) {
            stringstream ss;
            ss << "ERROR, NULL pointer in " << __FILE__ << ":" << __LINE__ << endl;
            cerr << ss.str();
         }
         #endif
         
         Real DX2     = cellPointers[c][0][CellParams::DX]; DX2 *= DX2;
         Real DY2     = cellPointers[c][0][CellParams::DY]; DY2 *= DY2;
         Real phi_111 = cellPointers[c][0][CellParams::PHI];
         Real rho_q   = cellPointers[c][0][CellParams::RHOQ_TOT];
         
         Real phi_011 = cellPointers[c][1][CellParams::PHI];
         Real phi_211 = cellPointers[c][2][CellParams::PHI];
         Real phi_101 = cellPointers[c][3][CellParams::PHI];
         Real phi_121 = cellPointers[c][4][CellParams::PHI];

         Real factor = 2*(1/DX2 + 1/DY2);
         Real rhs = ((phi_011+phi_211)/DX2 + (phi_101+phi_121)/DY2 + rho_q)/factor;
         Real correction = rhs - phi_111;
         cellPointers[c][0][CellParams::PHI] = phi_111 + weight*correction;
      }      
   }
   
   void PoissonSolverSOR::evaluate3D(std::vector<poisson::CellCache3D>& cellPointers,const int& cellColor) {
      
      const Real weight = 1.5;

      #pragma omp for
      for (size_t c=0; c<cellPointers.size(); ++c) {
         bool ok = true;
         #ifdef DEBUG_POISSON_SOR
         if (cellPointers[c][0] == NULL) ok = false;
         if (cellPointers[c][1] == NULL) ok = false;
         if (cellPointers[c][2] == NULL) ok = false;
         if (cellPointers[c][3] == NULL) ok = false;
         if (cellPointers[c][4] == NULL) ok = false;
         if (cellPointers[c][5] == NULL) ok = false;
         if (cellPointers[c][6] == NULL) ok = false;
         if (ok == false) {
            stringstream ss;
            ss << "ERROR, NULL pointer in " << __FILE__ << ":" << __LINE__ << endl;
            cerr << ss.str();
         }
         #endif
         
         Real DX2     = cellPointers[c][0][CellParams::DX]; DX2 *= DX2;
         Real DY2     = cellPointers[c][0][CellParams::DY]; DY2 *= DY2;
         Real DZ2     = cellPointers[c][0][CellParams::DZ]; DZ2 *= DZ2;
         Real phi_111 = cellPointers[c][0][CellParams::PHI];
         Real rho_q   = cellPointers[c][0][CellParams::RHOQ_TOT];
         
         Real phi_011 = cellPointers[c][1][CellParams::PHI];
         Real phi_211 = cellPointers[c][2][CellParams::PHI];
         Real phi_101 = cellPointers[c][3][CellParams::PHI];
         Real phi_121 = cellPointers[c][4][CellParams::PHI];
         Real phi_110 = cellPointers[c][5][CellParams::PHI];
         Real phi_112 = cellPointers[c][6][CellParams::PHI];

         Real factor = 2*(1/DX2 + 1/DY2 + 1/DZ2);
         Real rhs = ((phi_011+phi_211)/DX2 + (phi_101+phi_121)/DY2 + (phi_110+phi_112)/DZ2 + rho_q)/factor;
         Real correction = rhs - phi_111;
         cellPointers[c][0][CellParams::PHI] = phi_111 + weight*correction;
      }      
   }

   void PoissonSolverSOR::cachePointers2D(
               dccrg::Dccrg<spatial_cell::SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
               const std::vector<CellID>& cells,
               std::vector<poisson::CellCache3D>& redCache,
               std::vector<poisson::CellCache3D>& blackCache) {
      redCache.clear();
      blackCache.clear();

      for (size_t c=0; c<cells.size(); ++c) {
         // DO_NOT_COMPUTE cells are skipped
         if (mpiGrid[cells[c]]->sysBoundaryFlag == sysboundarytype::DO_NOT_COMPUTE) continue;

         // Calculate cell i/j/k indices
         dccrg::Types<3>::indices_t indices = mpiGrid.mapping.get_indices(cells[c]);

         CellCache3D cache;
         cache.cellID = cells[c];
         cache.cell = mpiGrid[cells[c]];
         cache[0]   = mpiGrid[cells[c]]->parameters;

         #ifdef DEBUG_POISSON_SOR
         if (cache.cell == NULL) {
            stringstream s;
            s << "ERROR, NULL pointer in " << __FILE__ << ":" << __LINE__ << endl;
            s << "\t Cell ID " << cells[c] << endl;
            cerr << s.str();
            exit(1);
         }
         #endif

         spatial_cell::SpatialCell* dummy = NULL;
         switch (mpiGrid[cells[c]]->sysBoundaryFlag) {
            case sysboundarytype::DO_NOT_COMPUTE:
               break;
            case sysboundarytype::NOT_SYSBOUNDARY:
               // Fetch pointers to this cell's (cell) parameters array, 
               // and pointers to +/- xyz face neighbors' arrays
               indices[0] -= 1; cache[1] = mpiGrid[ mpiGrid.mapping.get_cell_from_indices(indices,0) ]->parameters;
               indices[0] += 2; cache[2] = mpiGrid[ mpiGrid.mapping.get_cell_from_indices(indices,0) ]->parameters;
               indices[0] -= 1;
            
               indices[1] -= 1; cache[3] = mpiGrid[ mpiGrid.mapping.get_cell_from_indices(indices,0) ]->parameters;
               indices[1] += 2; cache[4] = mpiGrid[ mpiGrid.mapping.get_cell_from_indices(indices,0) ]->parameters;
               indices[1] -= 1;
               break;
               
            case sysboundarytype::ANTISYMMETRIC:
               if (indices[1] == 1 || indices[1] == Parameters::ycells_ini-2) {
                  // Get +/- x-neighbor pointers
                  indices[0] -= 1;
                  dummy = mpiGrid[ mpiGrid.mapping.get_cell_from_indices(indices,0) ];
                  if (dummy == NULL) cache[1] = bndryCellParams;
                  else               cache[1] = dummy->parameters;
                  indices[0] += 2;
                  dummy = mpiGrid[ mpiGrid.mapping.get_cell_from_indices(indices,0) ];
                  if (dummy == NULL) cache[2] = bndryCellParams;
                  else               cache[2] = dummy->parameters;
                  indices[0] -= 1;

                  // Set +/- y-neighbors both point to +y neighbor 
                  // if we are at the lower y-boundary, otherwise set both 
                  // y-neighbors point to -y neighbor.
                  if (indices[1] == 1) {
                     indices[1] += 1;
                     dummy = mpiGrid[ mpiGrid.mapping.get_cell_from_indices(indices,0) ];
                     if (dummy == NULL) {
                        cache[3] = bndryCellParams;
                        cache[4] = bndryCellParams;
                     } else {
                        cache[3] = dummy->parameters;
                        cache[4] = dummy->parameters;
                     }
                     indices[1] -= 1;
                  } else {
                     indices[1] -= 1;
                     dummy = mpiGrid[ mpiGrid.mapping.get_cell_from_indices(indices,0) ];
                     if (dummy == NULL) {
                        cache[3] = bndryCellParams;
                        cache[4] = bndryCellParams;
                     } else {
                        cache[3] = dummy->parameters;
                        cache[4] = dummy->parameters;
                     }
                     indices[1] += 1;
                  }
               }
               break;

            default:
               indices[0] -= 1;
               dummy = mpiGrid[ mpiGrid.mapping.get_cell_from_indices(indices,0) ];
               if (dummy == NULL) cache[1] = bndryCellParams;
               else               cache[1] = dummy->parameters;
               indices[0] += 2;
               dummy = mpiGrid[ mpiGrid.mapping.get_cell_from_indices(indices,0) ];
               if (dummy == NULL) cache[2] = bndryCellParams;
               else               cache[2] = dummy->parameters;
               indices[0] -= 1;

               indices[1] -= 1; 
               dummy = mpiGrid[ mpiGrid.mapping.get_cell_from_indices(indices,0) ];
               if (dummy == NULL) cache[3] = bndryCellParams;
               else               cache[3] = dummy->parameters;
               indices[1] += 2;
               dummy = mpiGrid[ mpiGrid.mapping.get_cell_from_indices(indices,0) ];
               if (dummy == NULL) cache[4] = bndryCellParams;
               else               cache[4] = dummy->parameters;
               indices[1] -= 1;
               break;
         }

         if ((indices[0] + indices[1]%2 + indices[2]%2) % 2 == RED) {
            redCache.push_back(cache);
         } else {
            blackCache.push_back(cache);
         }
      } // for-loop over spatial cells
   }

   void PoissonSolverSOR::cachePointers3D(
            dccrg::Dccrg<spatial_cell::SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
            const std::vector<CellID>& cells,std::vector<poisson::CellCache3D>& redCache,
            std::vector<poisson::CellCache3D>& blackCache) {
      redCache.clear();
      blackCache.clear();
      
      for (size_t c=0; c<cells.size(); ++c) {
         // Calculate cell i/j/k indices
         dccrg::Types<3>::indices_t indices = mpiGrid.mapping.get_indices(cells[c]);

         if ((indices[0] + indices[1]%2 + indices[2]%2) % 2 == RED) {
            CellCache3D cache;

            // Cells on domain boundaries are not iterated
            if (mpiGrid[cells[c]]->sysBoundaryFlag != 1) continue;
            
            // Fetch pointers to this cell's (cell) parameters array, 
            // and pointers to +/- xyz face neighbors' arrays
            cache.cell = mpiGrid[cells[c]];
            cache[0] = mpiGrid[cells[c]]->parameters;
            
            indices[0] -= 1; cache[1] = mpiGrid[ mpiGrid.mapping.get_cell_from_indices(indices,0) ]->parameters;
            indices[0] += 2; cache[2] = mpiGrid[ mpiGrid.mapping.get_cell_from_indices(indices,0) ]->parameters;
            indices[0] -= 1;
            
            indices[1] -= 1; cache[3] = mpiGrid[ mpiGrid.mapping.get_cell_from_indices(indices,0) ]->parameters;
            indices[1] += 2; cache[4] = mpiGrid[ mpiGrid.mapping.get_cell_from_indices(indices,0) ]->parameters;
            indices[1] -= 1;
            
            indices[2] -= 1; cache[5] = mpiGrid[ mpiGrid.mapping.get_cell_from_indices(indices,0) ]->parameters;
            indices[2] += 2; cache[6] = mpiGrid[ mpiGrid.mapping.get_cell_from_indices(indices,0) ]->parameters;
            indices[2] -= 1;
            
            redCache.push_back(cache);
         } else {
            CellCache3D cache;
            
            // Cells on domain boundaries are not iterated
            if (mpiGrid[cells[c]]->sysBoundaryFlag != 1) continue;
            
            // Fetch pointers to this cell's (cell) parameters array,
            // and pointers to +/- xyz face neighbors' arrays
            cache.cell = mpiGrid[cells[c]];
            cache[0] = mpiGrid[cells[c]]->parameters;
            
            indices[0] -= 1; cache[1] = mpiGrid[ mpiGrid.mapping.get_cell_from_indices(indices,0) ]->parameters;
            indices[0] += 2; cache[2] = mpiGrid[ mpiGrid.mapping.get_cell_from_indices(indices,0) ]->parameters;
            indices[0] -= 1;
            
            indices[1] -= 1; cache[3] = mpiGrid[ mpiGrid.mapping.get_cell_from_indices(indices,0) ]->parameters;
            indices[1] += 2; cache[4] = mpiGrid[ mpiGrid.mapping.get_cell_from_indices(indices,0) ]->parameters;
            indices[1] -= 1;
            
            indices[2] -= 1; cache[5] = mpiGrid[ mpiGrid.mapping.get_cell_from_indices(indices,0) ]->parameters;
            indices[2] += 2; cache[6] = mpiGrid[ mpiGrid.mapping.get_cell_from_indices(indices,0) ]->parameters;
            indices[2] -= 1;
            
            blackCache.push_back(cache);
         }
      }
   }

   bool PoissonSolverSOR::solve(dccrg::Dccrg<spatial_cell::SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid) {
      bool success = true;

      // If mesh partitioning has changed, recalculate pointer caches
      if (Parameters::meshRepartitioned == true) {
         phiprof::start("Pointer Caching");
         if (Poisson::is2D == true) {
            cachePointers2D(mpiGrid,mpiGrid.get_local_cells_on_process_boundary(POISSON_NEIGHBORHOOD_ID),bndryCellPointersRED,bndryCellPointersBLACK);
            cachePointers2D(mpiGrid,mpiGrid.get_local_cells_not_on_process_boundary(POISSON_NEIGHBORHOOD_ID),innerCellPointersRED,innerCellPointersBLACK);
         } else {
            cachePointers3D(mpiGrid,mpiGrid.get_local_cells_on_process_boundary(POISSON_NEIGHBORHOOD_ID),bndryCellPointersRED,bndryCellPointersBLACK);
            cachePointers3D(mpiGrid,mpiGrid.get_local_cells_not_on_process_boundary(POISSON_NEIGHBORHOOD_ID),innerCellPointersRED,innerCellPointersBLACK);
         }
         phiprof::stop("Pointer Caching");
      }

      // Calculate charge density
#warning CHANGE ME after DCCRG works
      //phiprof::start("MPI (RHOQ)");
      SpatialCell::set_mpi_transfer_type(Transfer::CELL_RHOQ_TOT,false);
      //mpiGrid.start_remote_neighbor_copy_receives(POISSON_NEIGHBORHOOD_ID);
      //phiprof::stop("MPI (RHOQ)");
      for (size_t c=0; c<bndryCellPointersRED.size(); ++c) calculateChargeDensity(bndryCellPointersRED[c].cell);
      for (size_t c=0; c<bndryCellPointersBLACK.size(); ++c) calculateChargeDensity(bndryCellPointersBLACK[c].cell);
      //phiprof::start("MPI (RHOQ)");
      //mpiGrid.start_remote_neighbor_copy_sends(POISSON_NEIGHBORHOOD_ID);
      mpiGrid.start_remote_neighbor_copy_updates(POISSON_NEIGHBORHOOD_ID);
      //phiprof::stop("MPI (RHOQ)");
      for (size_t c=0; c<innerCellPointersRED.size(); ++c) calculateChargeDensity(innerCellPointersRED[c].cell);
      for (size_t c=0; c<innerCellPointersBLACK.size(); ++c) calculateChargeDensity(innerCellPointersBLACK[c].cell);
      phiprof::start("MPI (RHOQ)");
      //mpiGrid.wait_remote_neighbor_copy_updates(POISSON_NEIGHBORHOOD_ID);
      mpiGrid.wait_remote_neighbor_copy_updates(POISSON_NEIGHBORHOOD_ID);
      phiprof::stop("MPI (RHOQ)");

      #warning RED/BLACK pointers do not include boundary cells
      
      SpatialCell::set_mpi_transfer_type(Transfer::CELL_PHI,false);
      do {
         int iterations = 0;
         const int N_iterations = 10;

	 #pragma omp parallel
	   {
	      const int tid = omp_get_thread_num();
	      
	      // Iterate the potential N_iterations times and then
	      // check if the error is less than the required value
	      for (int N=0; N<N_iterations; ++N) {
		 // Make a copy of the potential if we are going 
		 // to evaluate the solution error
		 if (N == N_iterations-1) {
		    if (tid == 0) phiprof::start("Copy Old Potential");
		    #pragma omp for
		    for (size_t c=0; c<Poisson::localCellParams.size(); ++c) {
		       Poisson::localCellParams[c][CellParams::PHI_TMP] = Poisson::localCellParams[c][CellParams::PHI];
		    }
		    if (tid == 0) phiprof::stop("Copy Old Potential",Poisson::localCellParams.size(),"Spatial Cells");
		 }

		 // Solve red cells first, the black cells
		 if (solve(mpiGrid,RED  ) == false) success = false;
		 if (solve(mpiGrid,BLACK) == false) success = false;
	      }
	   } // #pragma omp parallel

         // Evaluate the error in potential solution and reiterate if necessary
         iterations += N_iterations;
         const Real relPotentialChange = error(mpiGrid);
         if (relPotentialChange <= Poisson::minRelativePotentialChange) break;
         if (iterations >= Poisson::maxIterations) break;
      } while (true);

      return success;
   }

   bool PoissonSolverSOR::solve(dccrg::Dccrg<spatial_cell::SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
                                const int& oddness) {
      // NOTE: This function is entered by all threads in OpenMP run,
      // so everything must be thread-safe!

      bool success = true;
      const int tid = omp_get_thread_num();

      #warning Always uses 2D solver at the moment
      //if (Poisson::is2D == true) evaluator = this->evaluate2D;
      //else                       evaluator = evaluate3D;
      
      // Compute new potential on process boundary cells
      if (tid == 0) phiprof::start("Evaluate potential");
      if (oddness == RED) evaluate2D(bndryCellPointersRED,oddness);
      else                evaluate2D(bndryCellPointersBLACK,oddness);      
      if (tid == 0) {
         size_t cells = bndryCellPointersRED.size() + bndryCellPointersBLACK.size();
	 phiprof::stop("Evaluate potential",cells,"Spatial Cells");

	 // Exchange new potential values on process boundaries
	 phiprof::start("MPI (start copy)");
	 mpiGrid.start_remote_neighbor_copy_updates(POISSON_NEIGHBORHOOD_ID);
	 phiprof::stop("MPI (start copy)");

	 phiprof::start("Evaluate potential");
      }

      // Compute new potential on inner cells
      if (oddness == RED) evaluate2D(innerCellPointersRED,oddness);
      else                evaluate2D(innerCellPointersBLACK,oddness);

      // Wait for MPI transfers to complete
      if (tid == 0) {
         size_t cells = innerCellPointersRED.size() + innerCellPointersBLACK.size();
	 phiprof::stop("Evaluate potential",cells,"Spatial Cells");
	 phiprof::start("MPI (wait copy)");
	 mpiGrid.wait_remote_neighbor_copy_updates(POISSON_NEIGHBORHOOD_ID);
	 phiprof::stop("MPI (wait copy)");
      }

      return success;
   }

} // namespace poisson
