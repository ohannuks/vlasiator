/*
This file is part of Vlasiator.

Copyright 2010, 2011, 2012, 2013 Finnish Meteorological Institute
*/

#include <boost/assign/list_of.hpp>
#include <cstdlib>
#include <iostream>
#include <iomanip> // for setprecision()
#include <cmath>
#include <vector>
#include <sstream>
#include <ctime>
#include "grid.h"
#include "vlasovmover.h"
#include "definitions.h"
#include "mpiconversion.h"
#include "logger.h"
#include "parameters.h"
#include "datareduction/datareducer.h"
#include "sysboundary/sysboundary.h"
#include "fieldsolver/fs_common.h"
#include "projects/project.h"
#include "iowrite.h"
#include "ioread.h"

#warning DEBUGGING can be removed
#include <vlsv_writer.h>

#ifdef PAPI_MEM
#include "papi.h" 
#endif 


using namespace std;
using namespace phiprof;

extern Logger logFile, diagnostic;

void initVelocityGridGeometry();
void initSpatialCellCoordinates(dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid);
void initializeStencils(dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid);

void initializeGrid(
   int argn,
   char **argc,
   dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
   SysBoundary& sysBoundaries,
   Project& project
) {
   int myRank;
   MPI_Comm_rank(MPI_COMM_WORLD,&myRank);
   
   // initialize velocity grid of spatial cells before creating cells in dccrg.initialize
   initVelocityGridGeometry();

   // Init Zoltan:
   float zoltanVersion;
   if (Zoltan_Initialize(argn,argc,&zoltanVersion) != ZOLTAN_OK) {
      if(myRank == MASTER_RANK) cerr << "\t ERROR: Zoltan initialization failed." << endl;
      exit(1);
   } else {
      logFile << "\t Zoltan " << zoltanVersion << " initialized successfully" << std::endl << writeVerbose;
   }
   
   MPI_Comm comm = MPI_COMM_WORLD;
   int neighborhood_size = max(FS_STENCIL_WIDTH, VLASOV_STENCIL_WIDTH); 

   const std::array<uint64_t, 3> grid_length = {{P::xcells_ini, P::ycells_ini, P::zcells_ini}};
   dccrg::Cartesian_Geometry::Parameters geom_params;
   geom_params.start[0] = P::xmin;
   geom_params.start[1] = P::ymin;
   geom_params.start[2] = P::zmin;
   geom_params.level_0_cell_length[0] = P::dx_ini;
   geom_params.level_0_cell_length[1] = P::dy_ini;
   geom_params.level_0_cell_length[2] = P::dz_ini;

   mpiGrid.initialize(
      grid_length,
      comm,
      &P::loadBalanceAlgorithm[0],
      neighborhood_size, // neighborhood size
      0, // maximum refinement level
      sysBoundaries.isBoundaryPeriodic(0),
      sysBoundaries.isBoundaryPeriodic(1),
      sysBoundaries.isBoundaryPeriodic(2)
   );
   mpiGrid.set_geometry(geom_params);
   
   initializeStencils(mpiGrid);

   mpiGrid.set_partitioning_option("IMBALANCE_TOL", P::loadBalanceTolerance);
   phiprof::start("Initial load-balancing");
   if (myRank == MASTER_RANK) logFile << "(INIT): Starting initial load balance." << endl << writeVerbose;
   mpiGrid.balance_load();
   recalculateLocalCellsCache();
   phiprof::stop("Initial load-balancing");
   
   if (myRank == MASTER_RANK) logFile << "(INIT): Set initial state." << endl << writeVerbose;
   phiprof::start("Set initial state");
   
   phiprof::start("Set spatial cell coordinates");
   initSpatialCellCoordinates(mpiGrid);
   phiprof::stop("Set spatial cell coordinates");
   
   phiprof::start("Initialize system boundary conditions");
   if(sysBoundaries.initSysBoundaries(project, P::t_min) == false) {
      if (myRank == MASTER_RANK) cerr << "Error in initialising the system boundaries." << endl;
      exit(1);
   }
   phiprof::stop("Initialize system boundary conditions");
   
   // Initialise system boundary conditions (they need the initialised positions!!)
   phiprof::start("Classify cells (sys boundary conditions)");
   if(sysBoundaries.classifyCells(mpiGrid) == false) {
      cerr << "(MAIN) ERROR: System boundary conditions were not set correctly." << endl;
      exit(1);
   }
   phiprof::stop("Classify cells (sys boundary conditions)");
   
   if (P::isRestart){
      logFile << "Restart from "<< P::restartFileName << std::endl << writeVerbose;
      phiprof::start("Read restart");
      if ( readGrid(mpiGrid,P::restartFileName)== false ){
         logFile << "(MAIN) ERROR: restarting failed"<<endl;
         exit(1);
      }
      phiprof::stop("Read restart");
      vector<uint64_t> cells = mpiGrid.get_cells();
      //set background field, FIXME should be read in from restart
#pragma omp parallel for schedule(dynamic)
      for (uint i=0; i<cells.size(); ++i) {
         SpatialCell* cell = mpiGrid[cells[i]];
         project.setCellBackgroundField(cell);
      }
   } else {
      //Initial state based on project, background field in all cells
      //and other initial values in non-sysboundary cells
      phiprof::start("Apply initial state");
      // Go through every cell on this node and initialize the 
      //    -Background field on all cells
      //  -Perturbed fields and ion distribution function in non-sysboundary cells
      //    Each initialization has to be independent to avoid threading problems 
      vector<uint64_t> cells = mpiGrid.get_cells();
      
      #pragma omp parallel for schedule(dynamic)
      for (uint i=0; i<cells.size(); ++i) {
         SpatialCell* cell = mpiGrid[cells[i]];
         project.setCellBackgroundField(cell);         
         if (cell->sysBoundaryFlag == sysboundarytype::NOT_SYSBOUNDARY) {
            project.setCell(cell);
         }
      }

      //initial state for sys-boundary cells
      phiprof::stop("Apply initial state");
      phiprof::start("Apply system boundary conditions state");
      if (sysBoundaries.applyInitialState(mpiGrid, project) == false) {
         cerr << " (MAIN) ERROR: System boundary conditions initial state was not applied correctly." << endl;
         exit(1);
      }      
      phiprof::stop("Apply system boundary conditions state");
      
      adjustVelocityBlocks(mpiGrid, cells, true);
      validateMesh(mpiGrid);
      shrink_to_fit_grid_data(mpiGrid); //get rid of excess data already here

      phiprof::start("Init moments");
      //compute moments, and set them  in RHO* and RHO_*_DT2. If restart, they are already read in
      calculateInitialVelocityMoments(mpiGrid);
      phiprof::stop("Init moments");
      
      /*set initial LB metric based on number of blocks, all others
       * will be based on time spent in acceleration*/
      for (uint i=0; i<cells.size(); ++i) {
         mpiGrid[cells[i]]->parameters[CellParams::LBWEIGHTCOUNTER] = mpiGrid[cells[i]]->get_number_of_velocity_blocks();
      }
   }
   
   //Balance load before we transfer all data below
   balanceLoad(mpiGrid, sysBoundaries);
   
   phiprof::initializeTimer("Fetch Neighbour data","MPI");
   phiprof::start("Fetch Neighbour data");
   // update complet cell spatial data for full stencil (
   SpatialCell::set_mpi_transfer_type(Transfer::ALL_SPATIAL_DATA);
   mpiGrid.update_copies_of_remote_neighbors(FULL_NEIGHBORHOOD_ID);
   phiprof::stop("Fetch Neighbour data");
   phiprof::stop("Set initial state");
}

// initialize velocity grid of spatial cells before creating cells in dccrg.initialize
void initVelocityGridGeometry(){
   Real meshLimits[6];
   meshLimits[0] = P::vxmin;
   meshLimits[1] = P::vxmax;
   meshLimits[2] = P::vymin;
   meshLimits[3] = P::vymax;
   meshLimits[4] = P::vzmin;
   meshLimits[5] = P::vzmax;
   
   unsigned int gridLength[3];
   gridLength[0] = P::vxblocks_ini;
   gridLength[1] = P::vyblocks_ini;
   gridLength[2] = P::vzblocks_ini;
   
   unsigned int blockLength[3];
   blockLength[0] = block_vx_length;
   blockLength[1] = block_vy_length;
   blockLength[2] = block_vz_length;

   spatial_cell::SpatialCell::initialize_mesh(meshLimits,gridLength,blockLength,P::sparseMinValue,P::amrMaxVelocityRefLevel);
}

void initSpatialCellCoordinates(dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid) {
   vector<uint64_t> cells = mpiGrid.get_cells();
   #pragma omp parallel for
   for (uint i=0; i<cells.size(); ++i) {
      std::array<double, 3> cell_min = mpiGrid.geometry.get_min(cells[i]);
      std::array<double, 3> cell_length = mpiGrid.geometry.get_length(cells[i]);
      
      mpiGrid[cells[i]]->parameters[CellParams::XCRD] = cell_min[0];
      mpiGrid[cells[i]]->parameters[CellParams::YCRD] = cell_min[1];
      mpiGrid[cells[i]]->parameters[CellParams::ZCRD] = cell_min[2];
      mpiGrid[cells[i]]->parameters[CellParams::DX  ] = cell_length[0];
      mpiGrid[cells[i]]->parameters[CellParams::DY  ] = cell_length[1];
      mpiGrid[cells[i]]->parameters[CellParams::DZ  ] = cell_length[2];
   }
}


void balanceLoad(dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid, SysBoundary& sysBoundaries){
   // Invalidate cached cell lists
   Parameters::meshRepartitioned = true;

   // tell other processes which velocity blocks exist in remote spatial cells
   phiprof::initializeTimer("Balancing load", "Load balance");
   phiprof::start("Balancing load");

   phiprof::start("deallocate boundary data");
   //deallocate blocks in remote cells to decrease memory load
   deallocateRemoteCellBlocks(mpiGrid);

   phiprof::stop("deallocate boundary data");
   //set weights based on each cells LB weight counter
   vector<uint64_t> cells = mpiGrid.get_cells();
   for (uint i=0; i<cells.size(); ++i){
      //Set weight. If acceleration is enabled then we use the weight
      //counter which is updated in acceleration, otherwise we just
      //use the number of blocks.
      if (P::propagateVlasovAcceleration) 
         mpiGrid.set_cell_weight(cells[i], mpiGrid[cells[i]]->parameters[CellParams::LBWEIGHTCOUNTER]);
      else
         mpiGrid.set_cell_weight(cells[i], mpiGrid[cells[i]]->get_number_of_velocity_blocks());
      //reset counter
      mpiGrid[cells[i]]->parameters[CellParams::LBWEIGHTCOUNTER] = 0.0;
   }
   phiprof::start("dccrg.initialize_balance_load");
   mpiGrid.initialize_balance_load(true);
   phiprof::stop("dccrg.initialize_balance_load");
   
   const std::unordered_set<uint64_t>& incoming_cells = mpiGrid.get_cells_added_by_balance_load();
   std::vector<uint64_t> incoming_cells_list (incoming_cells.begin(),incoming_cells.end()); 

   const std::unordered_set<uint64_t>& outgoing_cells = mpiGrid.get_cells_removed_by_balance_load();
   std::vector<uint64_t> outgoing_cells_list (outgoing_cells.begin(),outgoing_cells.end()); 
   
   /*transfer cells in parts to preserve memory*/
   phiprof::start("Data transfers");
   const uint64_t num_part_transfers=5;
   for (uint64_t transfer_part=0; transfer_part<num_part_transfers; transfer_part++) {
      //Set transfers on/off for the incming cells in this transfer set and prepare for receive
      for (unsigned int i=0;i<incoming_cells_list.size();i++){
         uint64_t cell_id=incoming_cells_list[i];
         SpatialCell* cell = mpiGrid[cell_id];
         if (cell_id%num_part_transfers!=transfer_part) {
            cell->set_mpi_transfer_enabled(false);
         } else {
            cell->set_mpi_transfer_enabled(true);
         }
      }
      
      //Set transfers on/off for the outgoing cells in this transfer set
      for (unsigned int i=0; i<outgoing_cells_list.size(); i++) {
         uint64_t cell_id=outgoing_cells_list[i];
         SpatialCell* cell = mpiGrid[cell_id];
         if (cell_id%num_part_transfers!=transfer_part) {
            cell->set_mpi_transfer_enabled(false);
         } else {
            cell->set_mpi_transfer_enabled(true);
         }
      }

      //Transfer velocity block list
      SpatialCell::set_mpi_transfer_type(Transfer::VEL_BLOCK_LIST_STAGE1);
      mpiGrid.continue_balance_load();
      SpatialCell::set_mpi_transfer_type(Transfer::VEL_BLOCK_LIST_STAGE2);
      mpiGrid.continue_balance_load();
      
      for (unsigned int i=0; i<incoming_cells_list.size(); i++) {
         uint64_t cell_id=incoming_cells_list[i];
         SpatialCell* cell = mpiGrid[cell_id];
         if (cell_id%num_part_transfers==transfer_part) {
            phiprof::start("Preparing receives");
            // reserve space for velocity block data in arriving remote cells
            cell->prepare_to_receive_blocks();
            phiprof::stop("Preparing receives", incoming_cells_list.size(), "Spatial cells");
         }
      }

      //do the actual transfer of data for the set of cells to be transferred
      phiprof::start("transfer_all_data");
      SpatialCell::set_mpi_transfer_type(Transfer::ALL_DATA);
      mpiGrid.continue_balance_load();
      phiprof::stop("transfer_all_data");
      
      //Free memory for cells that have been sent (the blockdata)
      for (unsigned int i=0;i<outgoing_cells_list.size();i++){
         uint64_t cell_id=outgoing_cells_list[i];
         SpatialCell* cell = mpiGrid[cell_id];
         if (cell_id%num_part_transfers==transfer_part)
         cell->clear(); //free memory of this cell as it has already been transferred. It will not be used anymore
      }
   }
   phiprof::stop("Data transfers");
   
   //finish up load balancing
   phiprof::start("dccrg.finish_balance_load");
   mpiGrid.finish_balance_load();
   phiprof::stop("dccrg.finish_balance_load");

   //Make sure transfers are enabled for all cells
   recalculateLocalCellsCache();
   cells = mpiGrid.get_cells();
   for (uint i=0; i<cells.size(); ++i) mpiGrid[cells[i]]->set_mpi_transfer_enabled(true);

   // Communicate all spatial data for FULL neighborhood, which
   // includes all data with the exception of dist function data
   SpatialCell::set_mpi_transfer_type(Transfer::ALL_SPATIAL_DATA);
   mpiGrid.update_copies_of_remote_neighbors(FULL_NEIGHBORHOOD_ID);

   phiprof::start("update block lists");
   //new partition, re/initialize blocklists of remote cells.
   updateRemoteVelocityBlockLists(mpiGrid);
   phiprof::stop("update block lists");

   phiprof::start("update sysboundaries");
   sysBoundaries.updateSysBoundariesAfterLoadBalance( mpiGrid );
   phiprof::stop("update sysboundaries");

   phiprof::start("Init solvers");
   // Initialize field propagator (only if in use):
   if (Parameters::propagateField == true) {
      if (initializeFieldPropagatorAfterRebalance(mpiGrid) == false) {
         logFile << "(MAIN): Field propagator did not initialize correctly!" << endl << writeVerbose;
         exit(1);
      }
   }

   phiprof::stop("Init solvers");   
   phiprof::stop("Balancing load");
}

/*
  Adjust sparse velocity space to make it consistent in all 6 dimensions.

  Further documentation in grid.h
*/
bool adjustVelocityBlocks(dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
                          const vector<uint64_t>& cellsToAdjust,
                          bool doPrepareToReceiveBlocks) {
   phiprof::initializeTimer("re-adjust blocks","Block adjustment");
   phiprof::start("re-adjust blocks");
   const vector<CellID>& cells = getLocalCells();

   phiprof::start("Compute with_content_list");
   #pragma omp parallel for  
   for (uint i=0; i<cells.size(); ++i) {
      mpiGrid[cells[i]]->updateSparseMinValue();
      mpiGrid[cells[i]]->update_velocity_block_content_lists();
   }
   phiprof::stop("Compute with_content_list");

   phiprof::initializeTimer("Transfer with_content_list","MPI");
   phiprof::start("Transfer with_content_list");
   SpatialCell::set_mpi_transfer_type(Transfer::VEL_BLOCK_WITH_CONTENT_STAGE1 );
   mpiGrid.update_copies_of_remote_neighbors(NEAREST_NEIGHBORHOOD_ID);
   SpatialCell::set_mpi_transfer_type(Transfer::VEL_BLOCK_WITH_CONTENT_STAGE2 );
   mpiGrid.update_copies_of_remote_neighbors(NEAREST_NEIGHBORHOOD_ID);
   phiprof::stop("Transfer with_content_list");
   
   //Adjusts velocity blocks in local spatial cells, doesn't adjust velocity blocks in remote cells.

   phiprof::start("Adjusting blocks");
#pragma omp parallel for schedule(dynamic)
   for (unsigned int i=0; i<cellsToAdjust.size(); ++i) {
      Real density_pre_adjust=0.0;
      Real density_post_adjust=0.0;
      uint64_t cell_id=cellsToAdjust[i];
      SpatialCell* cell = mpiGrid[cell_id];
      
      // gather spatial neighbor list and create vector with pointers to neighbor spatial cells
      const vector<uint64_t>* neighbors = mpiGrid.get_neighbors_of(cell_id, NEAREST_NEIGHBORHOOD_ID);
      vector<SpatialCell*> neighbor_ptrs;
      neighbor_ptrs.reserve(neighbors->size());
      for (vector<uint64_t>::const_iterator neighbor_id = neighbors->begin(); neighbor_id != neighbors->end(); ++neighbor_id) {
         if (*neighbor_id == 0 || *neighbor_id == cell_id) {
            continue;
         }
         neighbor_ptrs.push_back(mpiGrid[*neighbor_id]);
      }
      if (P::sparse_conserve_mass) {
         for (size_t i=0; i<cell->get_number_of_velocity_blocks()*WID3; ++i) {
            density_pre_adjust += cell->get_data()[i];
         }
      }
      cell->adjust_velocity_blocks(neighbor_ptrs);

      if (P::sparse_conserve_mass) {
         for (size_t i=0; i<cell->get_number_of_velocity_blocks()*WID3; ++i) {
            density_post_adjust += cell->get_data()[i];
         }
         if (density_post_adjust != 0.0) {
            for (size_t i=0; i<cell->get_number_of_velocity_blocks()*WID3; ++i) {
               cell->get_data()[i] *= density_pre_adjust/density_post_adjust;
            }
         }
      }
   }
   
   phiprof::stop("Adjusting blocks");
   //Updated newly adjusted velocity block lists on remote cells, and
   //prepare to receive block data
   if(doPrepareToReceiveBlocks) {
      updateRemoteVelocityBlockLists(mpiGrid);
   }
   phiprof::stop("re-adjust blocks");
   return true;
}

/*! Shrink to fit velocity space data to save memory.
 * \param mpiGrid Spatial grid
 */
void shrink_to_fit_grid_data(dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid) {
   const std::vector<CellID>& cells = getLocalCells();
   #pragma omp parallel for
   for(size_t i=0; i<cells.size(); ++i) {
      mpiGrid[cells[i]]->shrink_to_fit();
   }

   const std::vector<CellID> remote_cells = mpiGrid.get_remote_cells_on_process_boundary(DIST_FUNC_NEIGHBORHOOD_ID);
   #pragma omp parallel for
   for(size_t i=0; i<remote_cells.size(); ++i) {
      mpiGrid[remote_cells[i]]->shrink_to_fit();
   }
}

/*! Estimates memory consumption and writes it into logfile. Collective operation on MPI_COMM_WORLD
 * \param mpiGrid Spatial grid
 */
void report_grid_memory_consumption(dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid) {
   /*now report memory consumption into logfile*/
   const vector<CellID>& cells = getLocalCells();
   const std::vector<uint64_t> remote_cells = mpiGrid.get_remote_cells_on_process_boundary();   
   int rank,n_procs;
   MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   /* Compute memory statistics of the memory consumption of the spatial cells.
    * Internally we use double as MPI does
    * not define proper uint64_t datatypes for MAXLOCNot Real, as we
    * want double here not to loose accuracy.
    */

   /*report data for memory needed by blocks*/
   double mem[6] = {0};
   double sum_mem[6];
   
   for(unsigned int i=0;i<cells.size();i++){
      mem[0] += mpiGrid[cells[i]]->get_cell_memory_size();
      mem[3] += mpiGrid[cells[i]]->get_cell_memory_capacity();
   }

   for(unsigned int i=0;i<remote_cells.size();i++){
      mem[1] += mpiGrid[remote_cells[i]]->get_cell_memory_size();
      mem[4] += mpiGrid[remote_cells[i]]->get_cell_memory_capacity();
   }
   
   mem[2] = mem[0] + mem[1];//total meory according to size()
   mem[5] = mem[3] + mem[4];//total memory according to capacity()


   MPI_Reduce(mem, sum_mem, 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

   logFile << "(MEM) Total size: " << sum_mem[2] << endl;   
   logFile << "(MEM) Total capacity " << sum_mem[5] << endl;   
   
   struct {
      double val;
      int   rank;
   } max_mem[3],mem_usage_loc[3],min_mem[3];
   for(uint i = 0; i<3; i++){
      mem_usage_loc[i].val = mem[i + 3]; //report on capacity numbers (6: local cells, 7: remote cells, 8: all cells)
      mem_usage_loc[i].rank = rank;
   }
   
   MPI_Reduce(mem_usage_loc, max_mem, 3, MPI_DOUBLE_INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);
   MPI_Reduce(mem_usage_loc, min_mem, 3, MPI_DOUBLE_INT, MPI_MINLOC, 0, MPI_COMM_WORLD);
   
   logFile << "(MEM)   Average capacity: " << sum_mem[5]/n_procs << " local cells " << sum_mem[3]/n_procs << " remote cells " << sum_mem[4]/n_procs << endl;
   logFile << "(MEM)   Max capacity:     " << max_mem[2].val   << " on  process " << max_mem[2].rank << endl;
   logFile << "(MEM)   Min capacity:     " << min_mem[2].val   << " on  process " << min_mem[2].rank << endl;
   logFile << writeVerbose;
}

/*! Deallocates all block data in remote cells in order to save
 *  memory
 * \param mpiGrid Spatial grid
 */
void deallocateRemoteCellBlocks(dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid) {
   const std::vector<uint64_t> incoming_cells
      = mpiGrid.get_remote_cells_on_process_boundary(VLASOV_SOLVER_NEIGHBORHOOD_ID);
   for(unsigned int i=0;i<incoming_cells.size();i++){
      uint64_t cell_id=incoming_cells[i];
      SpatialCell* cell = mpiGrid[cell_id];
      if (cell != NULL) {
         cell->clear();
      }
   }

}

/*
Updates velocity block lists between remote neighbors and prepares local
copies of remote neighbors for receiving velocity block data.
*/
void updateRemoteVelocityBlockLists(dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid)
{
   // update velocity block lists For small velocity spaces it is
   // faster to do it in one operation, and not by first sending size,
   // then list. For large we do it in two steps
   phiprof::initializeTimer("Velocity block list update","MPI");
   phiprof::start("Velocity block list update");
   SpatialCell::set_mpi_transfer_type(Transfer::VEL_BLOCK_LIST_STAGE1);
   mpiGrid.update_copies_of_remote_neighbors(DIST_FUNC_NEIGHBORHOOD_ID);
   SpatialCell::set_mpi_transfer_type(Transfer::VEL_BLOCK_LIST_STAGE2);
   mpiGrid.update_copies_of_remote_neighbors(DIST_FUNC_NEIGHBORHOOD_ID);
   phiprof::stop("Velocity block list update");

   // Prepare spatial cells for receiving velocity block data
   phiprof::start("Preparing receives");
   const std::vector<uint64_t> incoming_cells
      = mpiGrid.get_remote_cells_on_process_boundary(DIST_FUNC_NEIGHBORHOOD_ID);
   #pragma omp parallel for
   for (unsigned int i=0; i<incoming_cells.size(); ++i) {
      uint64_t cell_id = incoming_cells[i];
      SpatialCell* cell = mpiGrid[cell_id];
      if (cell == NULL) {
         cerr << __FILE__ << ":" << __LINE__
              << " No data for spatial cell " << cell_id
              << endl;
         abort();
      }      
      cell->prepare_to_receive_blocks();
   }
   phiprof::stop("Preparing receives", incoming_cells.size(), "SpatialCells");
}

/*
  Set stencils. These are the stencils (in 2D, real ones in 3D of
  course). x are stencil neighbor to cell local cell o:

NEAREST FIELD_SOLVER  SYSBOUNDARIES  (nearest neighbor)
-----------
  xxx
  xox
  xxx
-----------

EXTENDED_SYSBOUNDARIES (second nearest neighbor, also in diagonal)
-----------
  xxxxx
  xxxxx
  xxoxx
  xxxxx
  xxxxx
-----------  

VLASOV
-----------  
    x
    x
  xxoxx
    x
    x
-----------    

VLASOV_{XYZ}
-----------
 xxoxxx
-----------

VLASOV_TARGET_{XYZ}
-----------
  xox

-----------

DIST_FUNC  (Includes all cells which should know about each others blocks and have space for them. VLASOV + SYSBOUNDARIES.
-----------  
    x
   xxx
  xxoxx
   xxx
    x
    
-----------    

   
FULL (Includes all possible communication)
-----------
  xxxxx
  xxxxx
  xxoxx
  xxxxx
  xxxxx

-----------

SHIFT_M_X    ox
SHIFT_P_X   xo
 Y, Z in the same way
*/

void initializeStencils(dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid){
   // set reduced neighborhoods
   typedef dccrg::Types<3>::neighborhood_item_t neigh_t;
   
   // set a reduced neighborhood for field solver
   std::vector<neigh_t> neighborhood;
   for (int z = -1; z <= 1; z++) {
      for (int y = -1; y <= 1; y++) {
         for (int x = -1; x <= 1; x++) {
            if (x == 0 && y == 0 && z == 0) {
               continue;
            }            
            neigh_t offsets = {{x, y, z}};
            neighborhood.push_back(offsets);
         }
      }
   }
   mpiGrid.add_neighborhood(FIELD_SOLVER_NEIGHBORHOOD_ID, neighborhood);
   mpiGrid.add_neighborhood(NEAREST_NEIGHBORHOOD_ID, neighborhood);
   mpiGrid.add_neighborhood(SYSBOUNDARIES_NEIGHBORHOOD_ID, neighborhood);

   neighborhood.clear();
   for (int z = -2; z <= 2; z++) {
      for (int y = -2; y <= 2; y++) {
         for (int x = -2; x <= 2; x++) {
            if (x == 0 && y == 0 && z == 0) {
               continue;
            }
            neigh_t offsets = {{x, y, z}};
            neighborhood.push_back(offsets);
         }
      }
   }
   mpiGrid.add_neighborhood(SYSBOUNDARIES_EXTENDED_NEIGHBORHOOD_ID, neighborhood);

   /*add face neighbors if stencil width larger than 2*/
   for (int d = 3; d <= VLASOV_STENCIL_WIDTH; d++) {
      neighborhood.push_back({{ d, 0, 0}});
      neighborhood.push_back({{-d, 0, 0}});
      neighborhood.push_back({{0, d, 0}});
      neighborhood.push_back({{0,-d, 0}});
      neighborhood.push_back({{0, 0, d}});
      neighborhood.push_back({{0, 0,-d}});     
   }
   
   /*all possible communication pairs*/
   mpiGrid.add_neighborhood(FULL_NEIGHBORHOOD_ID, neighborhood);

   
   /*stencils for semilagrangian propagators*/ 
   neighborhood.clear();
   for (int d = -VLASOV_STENCIL_WIDTH; d <= VLASOV_STENCIL_WIDTH; d++) {
     if (d != 0) {
        neighborhood.push_back({{d, 0, 0}});
        neighborhood.push_back({{0, d, 0}});
        neighborhood.push_back({{0, 0, d}});
     }
   }
   mpiGrid.add_neighborhood(VLASOV_SOLVER_NEIGHBORHOOD_ID, neighborhood);

   // add remaining nearest neighbors for DIST_FUNC neighborhood
   for (int z = -1; z <= 1; z++) {
      for (int y = -1; y <= 1; y++) {
         for (int x = -1; x <= 1; x++) {
            //do not add cells already in neighborhood (vlasov solver)
            if (x == 0 && y == 0 ) continue;
            if (x == 0 && z == 0 ) continue;
            if (y == 0 && z == 0 ) continue;
            
            neigh_t offsets = {{x, y, z}};
            neighborhood.push_back(offsets);
         }
      }
   }
   mpiGrid.add_neighborhood(DIST_FUNC_NEIGHBORHOOD_ID, neighborhood);
   
   neighborhood.clear();
   for (int d = -VLASOV_STENCIL_WIDTH; d <= VLASOV_STENCIL_WIDTH; d++) {
     if (d != 0) {
        neighborhood.push_back({{d, 0, 0}});
     }
   }
   mpiGrid.add_neighborhood(VLASOV_SOLVER_X_NEIGHBORHOOD_ID, neighborhood);

   
   neighborhood.clear();
   for (int d = -VLASOV_STENCIL_WIDTH; d <= VLASOV_STENCIL_WIDTH; d++) {
     if (d != 0) {
        neighborhood.push_back({{0, d, 0}});
     }
   }
   mpiGrid.add_neighborhood(VLASOV_SOLVER_Y_NEIGHBORHOOD_ID, neighborhood);

   
   neighborhood.clear();
   for (int d = -VLASOV_STENCIL_WIDTH; d <= VLASOV_STENCIL_WIDTH; d++) {
     if (d != 0) {
        neighborhood.push_back({{0, 0, d}});
     }
   }
   mpiGrid.add_neighborhood(VLASOV_SOLVER_Z_NEIGHBORHOOD_ID, neighborhood);

   neighborhood.clear();
   for (int d = -1; d <= 1; d++) {
     if (d != 0) {
        neighborhood.push_back({{d, 0, 0}});
     }
   }
   mpiGrid.add_neighborhood(VLASOV_SOLVER_TARGET_X_NEIGHBORHOOD_ID, neighborhood);

   neighborhood.clear();
   for (int d = -1; d <= 1; d++) {
     if (d != 0) {
        neighborhood.push_back({{0, d, 0}});
     }
   }
   mpiGrid.add_neighborhood(VLASOV_SOLVER_TARGET_Y_NEIGHBORHOOD_ID, neighborhood);

   neighborhood.clear();
   for (int d = -1; d <= 1; d++) {
     if (d != 0) {
        neighborhood.push_back({{0, 0, d}});
     }
   }
   mpiGrid.add_neighborhood(VLASOV_SOLVER_TARGET_Z_NEIGHBORHOOD_ID, neighborhood);


   neighborhood.clear();
   neighborhood.push_back({{1, 0, 0}});
   mpiGrid.add_neighborhood(SHIFT_M_X_NEIGHBORHOOD_ID, neighborhood);
   neighborhood.clear();
   neighborhood.push_back({{0, 1, 0}});
   mpiGrid.add_neighborhood(SHIFT_M_Y_NEIGHBORHOOD_ID, neighborhood);
   neighborhood.clear();
   neighborhood.push_back({{0, 0, 1}});
   mpiGrid.add_neighborhood(SHIFT_M_Z_NEIGHBORHOOD_ID, neighborhood);
   neighborhood.clear();
   neighborhood.push_back({{-1, 0, 0}});
   mpiGrid.add_neighborhood(SHIFT_P_X_NEIGHBORHOOD_ID, neighborhood);
   neighborhood.clear();
   neighborhood.push_back({{0, -1, 0}});
   mpiGrid.add_neighborhood(SHIFT_P_Y_NEIGHBORHOOD_ID, neighborhood);
   neighborhood.clear();
   neighborhood.push_back({{0, 0, -1}});
   mpiGrid.add_neighborhood(SHIFT_P_Z_NEIGHBORHOOD_ID, neighborhood);
   
   // Add face neighbors, needed for Poisson solver
   neighborhood.clear();
   neighborhood.push_back({{-1, 0, 0}});
   neighborhood.push_back({{+1, 0, 0}});
   neighborhood.push_back({{ 0,-1, 0}});
   neighborhood.push_back({{ 0,+1, 0}});
   neighborhood.push_back({{ 0, 0,-1}});
   neighborhood.push_back({{ 0, 0,+1}});
   mpiGrid.add_neighborhood(POISSON_NEIGHBORHOOD_ID, neighborhood);
}

#warning This is for testing, can be removed later
void writeVelMesh(dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid) {
   const vector<CellID>& cells = getLocalCells();

   static int counter=0;
   
   stringstream fname;
   fname << "VelMesh.";
   fname.width(3);
   fname.fill(0);
   fname << counter << ".vlsv";
   
   vlsv::Writer vlsvWriter;
   vlsvWriter.open(fname.str(),MPI_COMM_WORLD,0,MPI_INFO_NULL);
   writeVelocityDistributionData(vlsvWriter,mpiGrid,cells,MPI_COMM_WORLD);   
   vlsvWriter.close();
   
   ++counter;
}

bool validateMesh(dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid) {
      bool rvalue = true;
      #ifndef AMR
         return rvalue;
      #endif

      bool internallyValid = false;
      
      // First make sure that all cells local to this process have a valid mesh.
      // After the mesh is internally valid, we will update mesh structures 
      // with remote neighbors for as many times as needed.
      //
      // Note that we still assume that each spatial cell has a valid mesh 
      // with respect to velocity neighbors, i.e., we only validate the mesh 
      // with respect to spatial neighbors here.
      if (internallyValid == false) {
         const vector<CellID>& cells = getLocalCells();
         int iter=0;
         
         do {
            // Iterate over all local spatial cells and calculate 
            // the necessary velocity block refinements
            vector<set<vmesh::GlobalID> > refinements(cells.size());
            
            for (size_t c=0; c<cells.size(); ++c) {
               SpatialCell* cell = mpiGrid[cells[c]];
               
               // Get all spatial neighbors
               const vector<CellID>* neighbors = mpiGrid.get_neighbors_of(cells[c],NEAREST_NEIGHBORHOOD_ID);
               
               // Iterate over all spatial neighbors
               for (size_t n=0; n<neighbors->size(); ++n) {
                  CellID nbrCellID = (*neighbors)[n];
                  const SpatialCell* nbr = mpiGrid[nbrCellID];
                  
                  // Iterate over all blocks in the spatial neighbor, 
                  // and check that the neighbor block does not have 
                  // existing grandparent in this cell
                  for (vmesh::LocalID b=0; b<nbr->get_number_of_velocity_blocks(); ++b) {
                     vmesh::GlobalID blockGID = nbr->get_velocity_block_global_id(b);
                     vmesh::GlobalID grandParentGID = cell->velocity_block_has_grandparent(blockGID);
                     if (grandParentGID != cell->invalid_global_id()) {
                        //cerr << "spatial nbr block " << blockGID << " has gparent " << grandParentGID << endl;
                        
                        refinements[c].insert(cell->get_velocity_block_parent(blockGID));
                     }
                  }                  
               }
            }
            
            if (iter == 0) writeVelMesh(mpiGrid);
            
            // Apply refinements
            for (size_t c=0; c<cells.size(); ++c) {
               if (refinements[c].size() > 0) {
                  cerr << "cell " << cells[c] << " needs " << refinements[c].size() << " refinements" << endl;
                  cerr << "\t current # blocks is " << mpiGrid[cells[c]]->get_number_of_velocity_blocks() << endl;
                  
                  
                  
               }
            }
            
            writeVelMesh(mpiGrid);
            ++iter;
            
            // Exit if none of the spatial cells needs to refine its mesh
            if (refinements.size() == 0) break;
            break;
         } while (true);
      }
      //exit(1);
         
      do {
         // Update remote neighbor velocity meshes. After the transfers complete,
         // mpi_number_of_blocks and mpi_velocity_block_list are valid. The mesh 
         // classes (vmesh) have not been updated yet, though.
         phiprof::start("Velocity block list update");
         SpatialCell::set_mpi_transfer_type(Transfer::VEL_BLOCK_LIST_STAGE1);
         mpiGrid.update_copies_of_remote_neighbors(DIST_FUNC_NEIGHBORHOOD_ID);
         SpatialCell::set_mpi_transfer_type(Transfer::VEL_BLOCK_LIST_STAGE2);
         mpiGrid.update_copies_of_remote_neighbors(DIST_FUNC_NEIGHBORHOOD_ID);
         phiprof::stop("Velocity block list update");

         int16_t meshChanged=0;
         
         // Figure out if we need to continue mesh updates.
         int16_t globalSuccess=0;
         MPI_Allreduce(&meshChanged,&globalSuccess,1,MPI_Type<int16_t>(),MPI_MAX,MPI_COMM_WORLD);
         if (globalSuccess == 0) break;
      } while (true);
      
      return rvalue;
   }
