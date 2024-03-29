/*
 * This file is part of Vlasiator.
 * 
 * Copyright 2010, 2011, 2012, 2013 Finnish Meteorological Institute
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 */

#include <cstdlib>
#include <mpi.h>
#include <iostream>
#include <limits>
#include <array>
#include "datareductionoperator.h"
//#include "../parameters.h"

using namespace std;

typedef Parameters P;

namespace DRO {
   
   // ************************************************************
   // ***** DEFINITIONS FOR DATAREDUCTIONOPERATOR BASE CLASS *****
   // ************************************************************
   
   /** DataReductionOperator base class constructor. The constructor is empty.*/
   DataReductionOperator::DataReductionOperator() { }
   
   /** DataReductionOperator base class virtual destructor. The destructor is empty.*/
   DataReductionOperator::~DataReductionOperator() { }
   
   /** Get info on the data the DataReductionOperator writes on disk. A DataReductionOperator writes 
    * an array on disk. Each element of the array is a vector with n elements. Finally, each 
    * vector element has a byte size, as given by the sizeof function.
    * @param dataType Basic datatype, must be int, uint, float
    * @param dataSize Byte size of written datatype, for example double-precision floating points
    * have byte size of sizeof(double).
    * @param vectorSize How many elements are in the vector returned by the DataReductionOperator.
    * @return If true, DataReductionOperator returned sensible values.
    */
   bool DataReductionOperator::getDataVectorInfo(std::string& dataType,unsigned int& dataSize,unsigned int& vectorSize) const {
      cerr << "ERROR: DataReductionOperator::getDataVectorInfo called insted of derived class function!" << endl;
      return false;
   }
   
   /** Get the name of the reduced data variable. The name is written to the disk as-is 
    * and is used in visualization.
    * @return The name of the data. The base class function returns an empty string.
    */
   std::string DataReductionOperator::getName() const {
      cerr << "ERROR: DataReductionOperator::getName called instead of derived class function!" << endl;
      return string("");
   }
   
   /** Get the name of the reduced data variable. The name is written to the disk as-is 
    * and is used in visualization.
    * @param population ID of the population
    * @return The name of the data. The base class function returns an empty string.
    */
   std::string DataReductionOperator::getName(const int population) const {
      cerr << "ERROR: DataReductionOperator::getName called instead of derived class function!" << endl;
      return string("");
   }
   
   // TODO update this documentation snippet.
   /** Reduce the data and write the data vector to the given buffer.
    * @param N_blocks Number of velocity blocks in array avgs.
    * @param avgs Array containing distribution function values for each velocity block.
    * @param blockParams Array containing the parameters of each velocity block.
    * @param buffer Buffer in which the reduced data is written.
    * @return If true, DataReductionOperator reduced data successfully.
    */
   bool DataReductionOperator::reduceData(const SpatialCell* cell,char* buffer) {
      cerr << "ERROR: DataReductionOperator::reduceData called instead of derived class function!" << endl;
      return false;
   }
   
   // TODO update this documentation snippet.
   /** Reduce the data and write the data vector to the given variable.
    * @param N_blocks Number of velocity blocks in array avgs.
    * @param avgs Array containing distribution function values for each velocity block.
    * @param blockParams Array containing the parameters of each velocity block.
    * @param result Real variable in which the reduced data is written.
    * @return If true, DataReductionOperator reduced data successfully.
    */
   bool DataReductionOperator::reduceData(const SpatialCell* cell,Real* result) {
      cerr << "ERROR: DataReductionOperator::reduceData called instead of derived class function!" << endl;
      return false;
   }
   
   // TODO update this documentation snippet.
   /** Reduce the data and write the data vector to the given buffer.
    * @param cell Spatial cell
    * @param population Population ID (see reducepopulation.cpp)
    * @param buffer Buffer where to save the results
    * @return If true, DataReductionOperator reduced data successfully.
    */
   bool DataReductionOperator::reduceData(const SpatialCell* cell, const int population, char* buffer) {
      cerr << "ERROR: DataReductionOperator::reduceData called instead of derived class function!" << endl;
      return false;
   }
   
   /** Set the SpatialCell whose data is going to be reduced by subsequent calls to 
    * DRO::DataReductionOperator::reduceData. This function is provided so that 
    * variables stored per SpatialCell can be accessed.
    * 
    * Spatial cell variables are stored in array SpatialCell::cpu_cellParams. 
    * The contents of array elements are stored in namespace CellParams. For example, 
    * cell.cpu_cellParams[%CellParams::EX] contains the electric field.
    * @param cell The SpatialCell whose data is to be reduced next.
    * @return If true, the SpatialCell was set correctly.
    */
   bool DataReductionOperator::setSpatialCell(const SpatialCell* cell) {
      cerr << "ERROR: DataReductionOperator::setSpatialCell called instead of derived class function!" << endl;
      return false;
   }
   
   
   
   
   
   DataReductionOperatorCellParams::DataReductionOperatorCellParams(const std::string& name,const unsigned int parameterIndex,const unsigned int vectorSize):
   DataReductionOperator() {
      _vectorSize=vectorSize;
      _name=name;
      _parameterIndex=parameterIndex;
   }
   DataReductionOperatorCellParams::~DataReductionOperatorCellParams() { }
   
   bool DataReductionOperatorCellParams::getDataVectorInfo(std::string& dataType,unsigned int& dataSize,unsigned int& vectorSize) const {
      dataType = "float";
      dataSize =  sizeof(Real);
      vectorSize = _vectorSize;
      return true;
   }
   
   std::string DataReductionOperatorCellParams::getName() const {return _name;}
   
   bool DataReductionOperatorCellParams::reduceData(const SpatialCell* cell,char* buffer) {
      const char* ptr = reinterpret_cast<const char*>(_data);
      for (uint i=0; i<_vectorSize*sizeof(Real); ++i){
         buffer[i] = ptr[i];
      }
      return true;
   }
   
   bool DataReductionOperatorCellParams::reduceData(const SpatialCell* cell,Real* buffer){
      //If _vectorSize is >1 it still works, we just give the first value and no other ones..
      *buffer=_data[0];
      return true;
   }
   bool DataReductionOperatorCellParams::setSpatialCell(const SpatialCell* cell) {
      if(std::isinf(cell->parameters[_parameterIndex]) || std::isnan(cell->parameters[_parameterIndex])) {
         string message = "The DataReductionOperator " + this->getName() + " returned a nan or an inf.";
         bailout(true, message, __FILE__, __LINE__);
      }
      _data  = &(cell->parameters[_parameterIndex]);
      return true;
   }




   
   DataReductionOperatorDerivatives::DataReductionOperatorDerivatives(const std::string& name,const unsigned int parameterIndex,const unsigned int vectorSize):
   DataReductionOperatorCellParams(name,parameterIndex,vectorSize) {

   }
   //a version with derivatives, this is the only function that is different
   bool DataReductionOperatorDerivatives::setSpatialCell(const SpatialCell* cell) {
      _data  = &(cell->derivatives[_parameterIndex]);
      return true;
   }


   DataReductionOperatorBVOLDerivatives::DataReductionOperatorBVOLDerivatives(const std::string& name,const unsigned int parameterIndex,const unsigned int vectorSize):
   DataReductionOperatorCellParams(name,parameterIndex,vectorSize) {
      
   }
   //a version with derivatives, this is the only function that is different
   bool DataReductionOperatorBVOLDerivatives::setSpatialCell(const SpatialCell* cell) {
      _data  = &(cell->derivativesBVOL[_parameterIndex]);
      return true;
   }
   
   
   
   //------------------ total BVOL --------------------------------------- 
   VariableBVol::VariableBVol(): DataReductionOperator() { }
   VariableBVol::~VariableBVol() { }
   
   bool VariableBVol::getDataVectorInfo(std::string& dataType,unsigned int& dataSize,unsigned int& vectorSize) const {
      dataType = "float";
      dataSize =  sizeof(Real);
      vectorSize = 3;
      return true;
   }
   
   std::string VariableBVol::getName() const {return "B_vol";}
   
   bool VariableBVol::reduceData(const SpatialCell* cell,char* buffer) {
      const char* ptr = reinterpret_cast<const char*>(B);
      for (uint i=0; i<3*sizeof(Real); ++i) buffer[i] = ptr[i];
      return true;
   }
   
   bool VariableBVol::setSpatialCell(const SpatialCell* cell) {
      B[0] = cell->parameters[CellParams::PERBXVOL] +  cell->parameters[CellParams::BGBXVOL];
      B[1] = cell->parameters[CellParams::PERBYVOL] +  cell->parameters[CellParams::BGBYVOL];
      B[2] = cell->parameters[CellParams::PERBZVOL] +  cell->parameters[CellParams::BGBZVOL];
      if(std::isinf(B[0]) || std::isnan(B[0]) ||
         std::isinf(B[1]) || std::isnan(B[1]) ||
         std::isinf(B[2]) || std::isnan(B[2])
      ) {
         string message = "The DataReductionOperator " + this->getName() + " returned a nan or an inf.";
         bailout(true, message, __FILE__, __LINE__);
      }
      return true;
   }




   //------------------ total B --------------------------------------- 
   VariableB::VariableB(): DataReductionOperator() { }
   VariableB::~VariableB() { }
   
   bool VariableB::getDataVectorInfo(std::string& dataType,unsigned int& dataSize,unsigned int& vectorSize) const {
      dataType = "float";
      dataSize =  sizeof(Real);
      vectorSize = 3;
      return true;
   }
   
   std::string VariableB::getName() const {return "B";}
   
   bool VariableB::reduceData(const SpatialCell* cell,char* buffer) {
      const char* ptr = reinterpret_cast<const char*>(B);
      for (uint i=0; i<3*sizeof(Real); ++i) buffer[i] = ptr[i];
      return true;
   }
   
   bool VariableB::setSpatialCell(const SpatialCell* cell) {
      B[0] = cell->parameters[CellParams::PERBX] +  cell->parameters[CellParams::BGBX];
      B[1] = cell->parameters[CellParams::PERBY] +  cell->parameters[CellParams::BGBY];
      B[2] = cell->parameters[CellParams::PERBZ] +  cell->parameters[CellParams::BGBZ];
      if(std::isinf(B[0]) || std::isnan(B[0]) ||
         std::isinf(B[1]) || std::isnan(B[1]) ||
         std::isinf(B[2]) || std::isnan(B[2])
      ) {
         string message = "The DataReductionOperator " + this->getName() + " returned a nan or an inf.";
         bailout(true, message, __FILE__, __LINE__);
      }
      return true;
   }
   
   
   //MPI rank
   MPIrank::MPIrank(): DataReductionOperator() { }
   MPIrank::~MPIrank() { }
   
   bool MPIrank::getDataVectorInfo(std::string& dataType,unsigned int& dataSize,unsigned int& vectorSize) const {
      dataType = "int";
      dataSize = 4;
      vectorSize = 1;
      return true;
   }
   
   std::string MPIrank::getName() const {return "MPI_rank";}
   
   bool MPIrank::reduceData(const SpatialCell* cell,char* buffer) {
      const char* ptr = reinterpret_cast<const char*>(&mpiRank);
      for (uint i=0; i<sizeof(int); ++i) buffer[i] = ptr[i];
      return true;
   }
   
   bool MPIrank::setSpatialCell(const SpatialCell* cell) {
      int intRank;
      MPI_Comm_rank(MPI_COMM_WORLD,&intRank);
      rank = 1.0*intRank;
      mpiRank = intRank;
      return true;
   }
   
   // BoundaryType
   BoundaryType::BoundaryType(): DataReductionOperator() { }
   BoundaryType::~BoundaryType() { }
   
   bool BoundaryType::getDataVectorInfo(std::string& dataType,unsigned int& dataSize,unsigned int& vectorSize) const {
      dataType = "int";
      dataSize = sizeof(int);
      vectorSize = 1;
      return true;
   }
   
   std::string BoundaryType::getName() const {return "Boundary_type";}
   
   bool BoundaryType::reduceData(const SpatialCell* cell,char* buffer) {
      const char* ptr = reinterpret_cast<const char*>(&boundaryType);
      for (uint i=0; i<sizeof(int); ++i) buffer[i] = ptr[i];
      return true;
   }
   
   bool BoundaryType::setSpatialCell(const SpatialCell* cell) {
      boundaryType = (int)cell->sysBoundaryFlag;
      return true;
   }


      // BoundaryLayer
   BoundaryLayer::BoundaryLayer(): DataReductionOperator() { }
   BoundaryLayer::~BoundaryLayer() { }
   
   bool BoundaryLayer::getDataVectorInfo(std::string& dataType,unsigned int& dataSize,unsigned int& vectorSize) const {
      dataType = "int";
      dataSize = sizeof(int);
      vectorSize = 1;
      return true;
   }
   
   std::string BoundaryLayer::getName() const {return "Boundary_layer";}
   
   bool BoundaryLayer::reduceData(const SpatialCell* cell,char* buffer) {
      const char* ptr = reinterpret_cast<const char*>(&boundaryLayer);
      for (uint i=0; i<sizeof(int); ++i) buffer[i] = ptr[i];
      return true;
   }
   
   bool BoundaryLayer::setSpatialCell(const SpatialCell* cell) {
      boundaryLayer = (int)cell->sysBoundaryLayer;
      return true;
   }
   
   // Blocks
   Blocks::Blocks(): DataReductionOperator() { }
   Blocks::~Blocks() { }
   
   bool Blocks::getDataVectorInfo(std::string& dataType,unsigned int& dataSize,unsigned int& vectorSize) const {
      dataType = "uint";
      dataSize = 4;
      vectorSize = 1;
      return true;
   }
   
   std::string Blocks::getName() const {return "Blocks";}
   
   bool Blocks::reduceData(const SpatialCell* cell,char* buffer) {
      const char* ptr = reinterpret_cast<const char*>(&nBlocks);
      for (uint i=0; i<sizeof(int); ++i) buffer[i] = ptr[i];
      return true;
   }
   
   bool Blocks::reduceData(const SpatialCell* cell,Real* buffer) {
      *buffer = 1.0 * nBlocks;
      return true;
   }
  
   bool Blocks::setSpatialCell(const SpatialCell* cell) {
      nBlocks = cell->get_number_of_velocity_blocks();
      return true;
   }
   
   
   // Scalar pressure 
   VariablePressure::VariablePressure(): DataReductionOperator() { }
   VariablePressure::~VariablePressure() { }
   
   std::string VariablePressure::getName() const {return "Pressure";}
   
   bool VariablePressure::getDataVectorInfo(std::string& dataType,unsigned int& dataSize,unsigned int& vectorSize) const {
      dataType = "float";
      dataSize =  sizeof(Real);
      vectorSize = 1;
      return true;
   }
   
   // Adding pressure calculations to Vlasiator.
   // p = m/3 * integral((v - <V>)^2 * f(r,v) dV), doing the sum of the x, y and z components.
   bool VariablePressure::reduceData(const SpatialCell* cell,char* buffer) {
      const Real HALF = 0.5;
      const Real THIRD = 1.0/3.0;
      # pragma omp parallel
      {
         Real thread_nvx2_sum = 0.0;
         Real thread_nvy2_sum = 0.0;
         Real thread_nvz2_sum = 0.0;
         # pragma omp for
         for (vmesh::LocalID n=0; n<cell->get_number_of_velocity_blocks(); n++) {
            const Real* parameters = cell->get_block_parameters(n);
            const Realf* block_data = cell->get_data(n);
            for (uint k=0; k<WID; ++k)
               for (uint j=0; j<WID; ++j)
                  for (uint i=0; i<WID; ++i) {
                     const Real VX = parameters[BlockParams::VXCRD] + (i+HALF) * parameters[BlockParams::DVX];
                     const Real VY = parameters[BlockParams::VYCRD] + (j+HALF) * parameters[BlockParams::DVY];
                     const Real VZ = parameters[BlockParams::VZCRD] + (k+HALF) * parameters[BlockParams::DVZ];
                     
                     const Real DV3 = parameters[BlockParams::DVX] * parameters[BlockParams::DVY] * parameters[BlockParams::DVZ];                   
                     thread_nvx2_sum += block_data[cellIndex(i,j,k)] * (VX - averageVX) * (VX - averageVX) * DV3;
                     thread_nvy2_sum += block_data[cellIndex(i,j,k)] * (VY - averageVY) * (VY - averageVY) * DV3;
                     thread_nvz2_sum += block_data[cellIndex(i,j,k)] * (VZ - averageVZ) * (VZ - averageVZ) * DV3;
            }
         }
         
         // Accumulate contributions coming from this velocity block to the 
         // spatial cell velocity moments. If multithreading / OpenMP is used, 
         // these updates need to be atomic:
         # pragma omp critical
         {
            Pressure += physicalconstants::MASS_PROTON * THIRD * (thread_nvx2_sum + thread_nvy2_sum + thread_nvz2_sum);
         }
      }
      const char* ptr = reinterpret_cast<const char*>(&Pressure);
      for (uint i=0; i<sizeof(Real); ++i) buffer[i] = ptr[i];
      return true;
   }

  bool VariablePressure::reduceData(const SpatialCell* cell,Real* buffer) {
    reduceData(cell,(char*)buffer);
    return true;
  }
   
   bool VariablePressure::setSpatialCell(const SpatialCell* cell) {
      if(cell-> parameters[CellParams::RHO] != 0.0) {
         averageVX = cell-> parameters[CellParams::RHOVX] / cell-> parameters[CellParams::RHO];
         averageVY = cell-> parameters[CellParams::RHOVY] / cell-> parameters[CellParams::RHO];
         averageVZ = cell-> parameters[CellParams::RHOVZ] / cell-> parameters[CellParams::RHO];
      } else {
         averageVX = 0.0;
         averageVY = 0.0;
         averageVZ = 0.0;
      }
      Pressure = 0.0;
      return true;
   }
   
   
   // Scalar pressure from the solvers
   VariablePressureSolver::VariablePressureSolver(): DataReductionOperator() { }
   VariablePressureSolver::~VariablePressureSolver() { }
   
   std::string VariablePressureSolver::getName() const {return "Pressure";}
   
   bool VariablePressureSolver::getDataVectorInfo(std::string& dataType,unsigned int& dataSize,unsigned int& vectorSize) const {
      dataType = "float";
      dataSize =  sizeof(Real);
      vectorSize = 1;
      return true;
   }
   
   bool VariablePressureSolver::reduceData(const SpatialCell* cell,char* buffer) {
      const char* ptr = reinterpret_cast<const char*>(&Pressure);
      for (uint i=0; i<sizeof(Real); ++i) buffer[i] = ptr[i];
      return true;
   }
   
   bool VariablePressureSolver::setSpatialCell(const SpatialCell* cell) {
      Pressure = 1.0/3.0 * (cell->parameters[CellParams::P_11] + cell->parameters[CellParams::P_22] + cell->parameters[CellParams::P_33]);
      return true;
   }
   
   
   // YK Adding pressure calculations to Vlasiator.
   // p_ij = m/3 * integral((v - <V>)_i(v - <V>)_j * f(r,v) dV)
   
   // Pressure tensor 6 components (11, 22, 33, 23, 13, 12) added by YK
   // Split into VariablePTensorDiagonal (11, 22, 33)
   // and VariablePTensorOffDiagonal (23, 13, 12)
   VariablePTensorDiagonal::VariablePTensorDiagonal(): DataReductionOperator() { }
   VariablePTensorDiagonal::~VariablePTensorDiagonal() { }
   
   std::string VariablePTensorDiagonal::getName() const {return "PTensorDiagonal";}
   
   bool VariablePTensorDiagonal::getDataVectorInfo(std::string& dataType,unsigned int& dataSize,unsigned int& vectorSize) const {
      dataType = "float";
      dataSize =  sizeof(Real);
      vectorSize = 3;
      return true;
   }
   
   bool VariablePTensorDiagonal::reduceData(const SpatialCell* cell,char* buffer) {
      const Real HALF = 0.5;
      # pragma omp parallel
      {
         Real thread_nvxvx_sum = 0.0;
         Real thread_nvyvy_sum = 0.0;
         Real thread_nvzvz_sum = 0.0;
         # pragma omp for
         for (vmesh::LocalID n=0; n<cell->get_number_of_velocity_blocks(); n++) {
            const Real* parameters = cell->get_block_parameters(n);
            const Realf* block_data = cell->get_data(n);
            for (uint k=0; k<WID; ++k)
               for (uint j=0; j<WID; ++j)
                  for (uint i=0; i<WID; ++i) {
                     const Real VX = parameters[BlockParams::VXCRD] + (i+HALF) * parameters[BlockParams::DVX];
                     const Real VY = parameters[BlockParams::VYCRD] + (j+HALF) * parameters[BlockParams::DVY];
                     const Real VZ = parameters[BlockParams::VZCRD] + (k+HALF) * parameters[BlockParams::DVZ];
                     const Real DV3 = parameters[BlockParams::DVX] * parameters[BlockParams::DVY] * parameters[BlockParams::DVZ];

                     thread_nvxvx_sum += block_data[cellIndex(i,j,k)] * (VX - averageVX) * (VX - averageVX) * DV3;
                     thread_nvyvy_sum += block_data[cellIndex(i,j,k)] * (VY - averageVY) * (VY - averageVY) * DV3;
                     thread_nvzvz_sum += block_data[cellIndex(i,j,k)] * (VZ - averageVZ) * (VZ - averageVZ) * DV3;
            }
         }
         
         // Accumulate contributions coming from this velocity block to the 
         // spatial cell velocity moments. If multithreading / OpenMP is used, 
         // these updates need to be atomic:
         # pragma omp critical
         {
            PTensor[0] += physicalconstants::MASS_PROTON * thread_nvxvx_sum;
            PTensor[1] += physicalconstants::MASS_PROTON * thread_nvyvy_sum;
            PTensor[2] += physicalconstants::MASS_PROTON * thread_nvzvz_sum;
         }
      }
      const char* ptr = reinterpret_cast<const char*>(&PTensor);
      for (uint i=0; i<3*sizeof(Real); ++i) buffer[i] = ptr[i];
      return true;
   }
   
   bool VariablePTensorDiagonal::setSpatialCell(const SpatialCell* cell) {
      if (cell-> parameters[CellParams::RHO] != 0.0) {
         averageVX = cell-> parameters[CellParams::RHOVX] / cell-> parameters[CellParams::RHO];
         averageVY = cell-> parameters[CellParams::RHOVY] / cell-> parameters[CellParams::RHO];
         averageVZ = cell-> parameters[CellParams::RHOVZ] / cell-> parameters[CellParams::RHO];
      } else {
         averageVX = 0.0;
         averageVY = 0.0;
         averageVZ = 0.0;
      }
      for(int i = 0; i < 3; i++) PTensor[i] = 0.0;
      return true;
   }
   
   VariablePTensorOffDiagonal::VariablePTensorOffDiagonal(): DataReductionOperator() { }
   VariablePTensorOffDiagonal::~VariablePTensorOffDiagonal() { }
   
   std::string VariablePTensorOffDiagonal::getName() const {return "PTensorOffDiagonal";}
   
   bool VariablePTensorOffDiagonal::getDataVectorInfo(std::string& dataType,unsigned int& dataSize,unsigned int& vectorSize) const {
      dataType = "float";
      dataSize =  sizeof(Real);
      vectorSize = 3;
      return true;
   }
   
   bool VariablePTensorOffDiagonal::reduceData(const SpatialCell* cell,char* buffer) {
      const Real HALF = 0.5;
      # pragma omp parallel
      {
         Real thread_nvxvy_sum = 0.0;
         Real thread_nvzvx_sum = 0.0;
         Real thread_nvyvz_sum = 0.0;
         # pragma omp for
         for (vmesh::LocalID n=0; n<cell->get_number_of_velocity_blocks(); n++) {
         const Real* parameters = cell->get_block_parameters(n);
         const Realf* block_data = cell->get_data(n);
            for (uint k=0; k<WID; ++k)
               for (uint j=0; j<WID; ++j)
                  for (uint i=0; i<WID; ++i) {
                     const Real VX = parameters[BlockParams::VXCRD] + (i+HALF) * parameters[BlockParams::DVX];
                     const Real VY = parameters[BlockParams::VYCRD] + (j+HALF) * parameters[BlockParams::DVY];
                     const Real VZ = parameters[BlockParams::VZCRD] + (k+HALF) * parameters[BlockParams::DVZ];
                     const Real DV3 = parameters[BlockParams::DVX] * parameters[BlockParams::DVY] * parameters[BlockParams::DVZ];
                     
                     thread_nvxvy_sum += block_data[cellIndex(i,j,k)] * (VX - averageVX) * (VY - averageVY) * DV3;
                     thread_nvzvx_sum += block_data[cellIndex(i,j,k)] * (VZ - averageVZ) * (VX - averageVX) * DV3;
                     thread_nvyvz_sum += block_data[cellIndex(i,j,k)] * (VY - averageVY) * (VZ - averageVZ) * DV3;
            }
         }
         
         // Accumulate contributions coming from this velocity block to the 
         // spatial cell velocity moments. If multithreading / OpenMP is used, 
         // these updates need to be atomic:
         # pragma omp critical
         {
            PTensor[0] += physicalconstants::MASS_PROTON * thread_nvyvz_sum;
            PTensor[1] += physicalconstants::MASS_PROTON * thread_nvzvx_sum;
            PTensor[2] += physicalconstants::MASS_PROTON * thread_nvxvy_sum;
         }
      }
      const char* ptr = reinterpret_cast<const char*>(&PTensor);
      for (uint i=0; i<3*sizeof(Real); ++i) buffer[i] = ptr[i];
      return true;
   }
   
   bool VariablePTensorOffDiagonal::setSpatialCell(const SpatialCell* cell) {
      if(cell-> parameters[CellParams::RHO] != 0.0) {
         averageVX = cell-> parameters[CellParams::RHOVX] / cell-> parameters[CellParams::RHO];
         averageVY = cell-> parameters[CellParams::RHOVY] / cell-> parameters[CellParams::RHO];
         averageVZ = cell-> parameters[CellParams::RHOVZ] / cell-> parameters[CellParams::RHO];
      } else {
         averageVX = 0.0;
         averageVY = 0.0;
         averageVZ = 0.0;
      }
      for(int i = 0; i < 3; i++) PTensor[i] = 0.0;
      return true;
   }
   
   // Integrated divergence of magnetic field
   // Integral of div B over the simulation volume =
   // Integral of flux of B on simulation volume surface
   DiagnosticFluxB::DiagnosticFluxB(): DataReductionOperator() { }
   DiagnosticFluxB::~DiagnosticFluxB() { }
   
   std::string DiagnosticFluxB::getName() const {return "FluxB";}
   
   bool DiagnosticFluxB::getDataVectorInfo(std::string& dataType,unsigned int& dataSize,unsigned int& vectorSize) const {
      dataType = "float";
      dataSize =  sizeof(Real);
      vectorSize = 1;
      return true;
   }
   
   bool DiagnosticFluxB::reduceData(const SpatialCell* cell,Real * result) {
      creal x = cell->parameters[CellParams::XCRD];
      creal dx = cell->parameters[CellParams::DX];
      creal y = cell->parameters[CellParams::YCRD];
      creal dy = cell->parameters[CellParams::DY];
      creal z = cell->parameters[CellParams::ZCRD];
      creal dz = cell->parameters[CellParams::DZ];
      creal cx = x + 0.5 * dx;
      creal cy = y + 0.5 * dy;
      creal cz = z + 0.5 * dz;
      
      Real value = 0.0;
      if(cx > Parameters::xmax - 2.0 * dx && cx < Parameters::xmax - dx) {
         value += cell->parameters[CellParams::PERBX];
      } else if (cx < Parameters::xmin + 2.0 * dx && cx > Parameters::xmin + dx) {
         value += -1.0*cell->parameters[CellParams::PERBX];
      }
      if(cy > Parameters::ymax - 2.0 * dy && cy < Parameters::ymax - dy) {
         value += cell->parameters[CellParams::PERBY];
      } else if (cy < Parameters::ymin + 2.0 * dy && cy > Parameters::ymin + dy) {
         value += -1.0*cell->parameters[CellParams::PERBY];
      }
      if(cz > Parameters::zmax - 2.0 * dz && cz < Parameters::zmax - dz) {
         value += cell->parameters[CellParams::PERBZ];
      } else if (cz < Parameters::zmin + 2.0 * dz && cz > Parameters::zmin + dz) {
         value += -1.0*cell->parameters[CellParams::PERBZ];
      }
      *result = value;
      
      return true;
   }
   
   bool DiagnosticFluxB::setSpatialCell(const SpatialCell* cell) {return true;}
   
   
   
   // YK Integrated divergence of electric field
   // Integral of div E over the simulation volume =
   // Integral of flux of E on simulation volume surface
   DiagnosticFluxE::DiagnosticFluxE(): DataReductionOperator() { }
   DiagnosticFluxE::~DiagnosticFluxE() { }
   
   std::string DiagnosticFluxE::getName() const {return "FluxE";}
   
   bool DiagnosticFluxE::getDataVectorInfo(std::string& dataType,unsigned int& dataSize,unsigned int& vectorSize) const {
      dataType = "float";
      dataSize =  sizeof(Real);
      vectorSize = 1;
      return true;
   }
   
   bool DiagnosticFluxE::reduceData(const SpatialCell* cell,Real * result) {
      creal x = cell->parameters[CellParams::XCRD];
      creal dx = cell->parameters[CellParams::DX];
      creal y = cell->parameters[CellParams::YCRD];
      creal dy = cell->parameters[CellParams::DY];
      creal z = cell->parameters[CellParams::ZCRD];
      creal dz = cell->parameters[CellParams::DZ];
      creal cx = x + 0.5 * dx;
      creal cy = y + 0.5 * dy;
      creal cz = z + 0.5 * dz;
      
      Real value = 0.0;
      if(cx > Parameters::xmax - 2.0 * dx && cx < Parameters::xmax - dx) {
         value += cell->parameters[CellParams::EX];
      } else if (cx < Parameters::xmin + 2.0 * dx && cx > Parameters::xmin + dx) {
         value += -1.0*cell->parameters[CellParams::EX];
      }
      if(cy > Parameters::ymax - 2.0 * dy && cy < Parameters::ymax - dy) {
         value += cell->parameters[CellParams::EY];
      } else if (cy < Parameters::ymin + 2.0 * dy && cy > Parameters::ymin + dy) {
         value += -1.0*cell->parameters[CellParams::EY];
      }
      if(cz > Parameters::zmax - 2.0 * dz && cz < Parameters::zmax - dz) {
         value += cell->parameters[CellParams::EZ];
      } else if (cz < Parameters::zmin + 2.0 * dz && cz > Parameters::zmin + dz) {
         value += -1.0*cell->parameters[CellParams::EZ];
      }
      *result = value;
      
      return true;
   }
   
   bool DiagnosticFluxE::setSpatialCell(const SpatialCell* cell) {return true;}
   
   // YK maximum value of the distribution function
   MaxDistributionFunction::MaxDistributionFunction(): DataReductionOperator() { }
   MaxDistributionFunction::~MaxDistributionFunction() { }
   
   std::string MaxDistributionFunction::getName() const {return "MaximumDistributionFunctionValue";}
   
   bool MaxDistributionFunction::getDataVectorInfo(std::string& dataType,unsigned int& dataSize,unsigned int& vectorSize) const {
      dataType = "float";
      dataSize =  sizeof(Real);
      vectorSize = 1;
      return true;
   }
   
   
   bool MaxDistributionFunction::reduceData(const SpatialCell* cell,Real* buffer) {
      const Real HALF = 0.5;
      
      #pragma omp parallel 
      {
         Real threadMax = std::numeric_limits<Real>::min();
         #pragma omp for
         for (vmesh::LocalID n=0; n<cell->get_number_of_velocity_blocks(); ++n) {
            const Realf* block_data = cell->get_data(n);
            for (uint k=0; k<WID; ++k)
               for (uint j=0; j<WID; ++j)
                  for (uint i=0; i<WID; ++i) {
                     const int celli=k*WID*WID+j*WID+i;
                     threadMax = max((Real)(block_data[celli]), threadMax);
            }
         }
         #pragma omp critical
         {
            maxF = max(threadMax, maxF);
         }
      }
      
      *buffer = maxF;
      return true;
   }
   
   bool MaxDistributionFunction::reduceData(const SpatialCell* cell,char* buffer) {
      Real dummy;
      reduceData(cell,&dummy);
      const char* ptr = reinterpret_cast<const char*>(&dummy);
      for (uint i=0; i<sizeof(Real); ++i) buffer[i] = ptr[i];
      return true;
   }
   
   bool MaxDistributionFunction::setSpatialCell(const SpatialCell* cell) {
      return true;
   }
   
   
   // YK minimum value of the distribution function
   MinDistributionFunction::MinDistributionFunction(): DataReductionOperator() { }
   MinDistributionFunction::~MinDistributionFunction() { }
   
   std::string MinDistributionFunction::getName() const {return "MinimumDistributionFunctionValue";}
   
   bool MinDistributionFunction::getDataVectorInfo(std::string& dataType,unsigned int& dataSize,unsigned int& vectorSize) const {
      dataType = "float";
      dataSize =  sizeof(Real);
      vectorSize = 1;
      return true;
   }
   
   
   bool MinDistributionFunction::reduceData(const SpatialCell* cell,Real* buffer) {
      const Real HALF = 0.5;
      
      #pragma omp parallel 
      {
         Real threadMin = std::numeric_limits<Real>::max();
         #pragma omp for
         for (vmesh::LocalID n=0; n<cell->get_number_of_velocity_blocks(); ++n) {
            const Realf* block_data = cell->get_data(n);
            for (uint k=0; k<WID; ++k)
               for (uint j=0; j<WID; ++j)
                  for (uint i=0; i<WID; ++i) {
                     const int celli=k*WID*WID+j*WID+i;
                     threadMin = min((Real)(block_data[celli]), threadMin);
            }
         }
         #pragma omp critical
         {
            minF = min(threadMin, minF);
         }
      }
      
      *buffer = minF;
      return true;
   }
   
   bool MinDistributionFunction::reduceData(const SpatialCell* cell,char* buffer) {
      Real dummy;
      reduceData(cell,&dummy);
      const char* ptr = reinterpret_cast<const char*>(&dummy);
      for (uint i=0; i<sizeof(Real); ++i) buffer[i] = ptr[i];
      return true;
   }
   
   bool MinDistributionFunction::setSpatialCell(const SpatialCell* cell) {
      return true;
   }

   //Helper function for getting the velocity cell ids that are a part of the backstream population:
   static void getBackstreamVelocityCells(const Real* block_parameters, vector<uint64_t> & vCellIds ) {
      const Real HALF = 0.5;
      // Go through every velocity cell (i, j, k are indices)
      for (uint k=0; k<WID; ++k) for (uint j=0; j<WID; ++j) for (uint i=0; i<WID; ++i) {
         // Get the vx, vy, vz coordinates of the velocity cell
         const Real VX = block_parameters[BlockParams::VXCRD] + (i+HALF) * block_parameters[BlockParams::DVX];
         const Real VY = block_parameters[BlockParams::VYCRD] + (j+HALF) * block_parameters[BlockParams::DVY];
         const Real VZ = block_parameters[BlockParams::VZCRD] + (k+HALF) * block_parameters[BlockParams::DVZ];
         // Compare the distance of the velocity cell from the center of the maxwellian distribution to the radius of the maxwellian distribution
         if( ( (P::backstreamvx - VX)*(P::backstreamvx - VX)
             + (P::backstreamvy - VY)*(P::backstreamvy - VY)
             + (P::backstreamvz - VZ)*(P::backstreamvz - VZ) )
             >
             P::backstreamradius*P::backstreamradius ) {
             //The velocity cell is a part of the backstream population:
             vCellIds.push_back(cellIndex(i,j,k));
          }
      }
   }
   //Helper function for getting the velocity cell ids that are a part of the backstream population:
   static void getNonBackstreamVelocityCells(const Real* block_parameters, vector<uint64_t> & vCellIds ) {
      const Real HALF = 0.5;
      for (uint k=0; k<WID; ++k) for (uint j=0; j<WID; ++j) for (uint i=0; i<WID; ++i) {
         const Real VX = block_parameters[BlockParams::VXCRD] + (i+HALF) * block_parameters[BlockParams::DVX];
         const Real VY = block_parameters[BlockParams::VYCRD] + (j+HALF) * block_parameters[BlockParams::DVY];
         const Real VZ = block_parameters[BlockParams::VZCRD] + (k+HALF) * block_parameters[BlockParams::DVZ];
         if( ( (P::backstreamvx - VX)*(P::backstreamvx - VX)
             + (P::backstreamvy - VY)*(P::backstreamvy - VY)
             + (P::backstreamvz - VZ)*(P::backstreamvz - VZ) )
             <=
             P::backstreamradius*P::backstreamradius ) {
             //The velocity cell is a part of the backstream population:
             vCellIds.push_back(cellIndex(i,j,k));
          }
      }
   }
   //Helper function for getting the velocity cell indices that are a part of the backstream population:
   static void getBackstreamVelocityCellIndices(
      const Real* block_parameters,
      vector<array<uint, 3>> & vCellIndices
   ) {
      const Real HALF = 0.5;
      // Go through a block's every velocity cell
      for (uint k=0; k<WID; ++k) for (uint j=0; j<WID; ++j) for (uint i=0; i<WID; ++i) {
         // Get the coordinates of the velocity cell (e.g. VX = block_vx_min_coordinates + (velocity_cell_indice_x+0.5)*length_of_velocity_cell_in_x_direction
         const Real VX = block_parameters[BlockParams::VXCRD] + (i+HALF) * block_parameters[BlockParams::DVX];
         const Real VY = block_parameters[BlockParams::VYCRD] + (j+HALF) * block_parameters[BlockParams::DVY];
         const Real VZ = block_parameters[BlockParams::VZCRD] + (k+HALF) * block_parameters[BlockParams::DVZ];
         // Calculate the distance of the velocity cell from the center of the maxwellian distribution and compare it to the approximate radius of the maxwellian distribution
         if( ( (P::backstreamvx - VX)*(P::backstreamvx - VX)
             + (P::backstreamvy - VY)*(P::backstreamvy - VY)
             + (P::backstreamvz - VZ)*(P::backstreamvz - VZ) )
             >
             P::backstreamradius*P::backstreamradius ) {
             //The velocity cell is a part of the backstream population because it is not within the radius:
             const array<uint, 3> indices{{i, j, k}};
             vCellIndices.push_back( indices );
          }
      }
   }
   //Helper function for getting the velocity cell indices that are not a part of the backstream population:
   static void getNonBackstreamVelocityCellIndices(const Real* block_parameters,
                                                 vector<array<uint, 3>> & vCellIndices ) {
      const Real HALF = 0.5;
      // Go through a block's every velocity cell
      for (uint k=0; k<WID; ++k) for (uint j=0; j<WID; ++j) for (uint i=0; i<WID; ++i) {
         // Get the coordinates of the velocity cell (e.g. VX = block_vx_min_coordinates + (velocity_cell_indice_x+0.5)*length_of_velocity_cell_in_x_direction
         const Real VX = block_parameters[BlockParams::VXCRD] + (i+HALF) * block_parameters[BlockParams::DVX];
         const Real VY = block_parameters[BlockParams::VYCRD] + (j+HALF) * block_parameters[BlockParams::DVY];
         const Real VZ = block_parameters[BlockParams::VZCRD] + (k+HALF) * block_parameters[BlockParams::DVZ];
         // Calculate the distance of the velocity cell from the center of the maxwellian distribution and compare it to the approximate radius of the maxwellian distribution
         if( ( (P::backstreamvx - VX)*(P::backstreamvx - VX)
             + (P::backstreamvy - VY)*(P::backstreamvy - VY)
             + (P::backstreamvz - VZ)*(P::backstreamvz - VZ) )
             <=
             P::backstreamradius*P::backstreamradius ) {
             //The velocity cell is a part of the backstream population because it is within the radius:
             const array<uint, 3> indices{{i, j, k}};
             vCellIndices.push_back( indices );
          }
      }
   }

   //Calculates rho backstream or rho non backstream
   static void rhoBackstreamCalculation( const SpatialCell * cell, const bool calculateBackstream, Real & rho ) {
      const Real HALF = 0.5;
      # pragma omp parallel
      {
         Real thread_n_sum = 0.0;
         # pragma omp for
         for (vmesh::LocalID n=0; n<cell->get_number_of_velocity_blocks(); ++n) {
            const Real* parameters = cell->get_block_parameters(n);
            const Realf* block_data = cell->get_data(n);
            const Real DV3 = parameters[BlockParams::DVX] * parameters[BlockParams::DVY] * parameters[BlockParams::DVZ];
            vector< uint64_t > vCells; //Velocity cell ids
            vCells.clear();
            if ( calculateBackstream == true ) {
               getBackstreamVelocityCells(parameters, vCells);
            } else {
               getNonBackstreamVelocityCells(parameters, vCells);
            }
            for( vector< uint64_t >::const_iterator it = vCells.begin(); it != vCells.end(); ++it ) {
               //velocity cell id = *it
               thread_n_sum += block_data[(*it)] * DV3;
            }
         }

         // Accumulate contributions coming from this velocity block
         // If multithreading / OpenMP is used, 
         // these updates need to be atomic:
         # pragma omp critical
         {
            rho += thread_n_sum;
         }
      }
      return;
   }

   static void rhoVBackstreamCalculation( const SpatialCell * cell, const bool calculateBackstream, Real * rhoV ) {
      const Real HALF = 0.5;
      // Make sure the rhoV is initialized
      rhoV[0] = 0;
      rhoV[1] = 0;
      rhoV[2] = 0;
      # pragma omp parallel
      {
         Real thread_nvx_sum = 0.0;
         Real thread_nvy_sum = 0.0;
         Real thread_nvz_sum = 0.0;
         # pragma omp for
         for (vmesh::LocalID n=0; n<cell->get_number_of_velocity_blocks(); ++n) {
            const Real* parameters = cell->get_block_parameters(n);
            const Realf* block_data = cell->get_data(n);
            const Real DV3 = parameters[BlockParams::DVX] * parameters[BlockParams::DVY] * parameters[BlockParams::DVZ]; // Get the volume of a velocity cell
            // Get the velocity cell indices of the cells that are a part of the backstream population
            vector< array<uint, 3> > vCellIndices;
            vCellIndices.clear();
            // Save indices to the std::vector
            if( calculateBackstream == true ) {
               getBackstreamVelocityCellIndices(parameters, vCellIndices);
            } else {
               getNonBackstreamVelocityCellIndices(parameters, vCellIndices);
            }
            // We have now fethced all of the needed velocity cell indices, so now go through them:
            for( vector< array<uint, 3> >::const_iterator it = vCellIndices.begin(); it != vCellIndices.end(); ++it ) {
               // Get the indices of the current iterated velocity cell
               const array<uint, 3> indices = *it;
               const uint i = indices[0];
               const uint j = indices[1];
               const uint k = indices[2];
               // Get the coordinates of the velocity cell (e.g. VX = block_vx_min_coordinates + (velocity_cell_indice_x+0.5)*length_of_velocity_cell_in_x_direction)
               const Real VX = parameters[BlockParams::VXCRD] + (i+HALF) * parameters[BlockParams::DVX];
               const Real VY = parameters[BlockParams::VYCRD] + (j+HALF) * parameters[BlockParams::DVY];
               const Real VZ = parameters[BlockParams::VZCRD] + (k+HALF) * parameters[BlockParams::DVZ];
               // Add the value of the coordinates and multiply by the AVGS value of the velocity cell and the volume of the velocity cell
               thread_nvx_sum += block_data[cellIndex(i,j,k)]*VX*DV3;
               thread_nvy_sum += block_data[cellIndex(i,j,k)]*VY*DV3;
               thread_nvz_sum += block_data[cellIndex(i,j,k)]*VZ*DV3;
            }
         }

         // Accumulate contributions coming from this velocity block
         // If multithreading / OpenMP is used, 
         // these updates need to be atomic:
         # pragma omp critical
         {
            rhoV[0] += thread_nvx_sum;
            rhoV[1] += thread_nvy_sum;
            rhoV[2] += thread_nvz_sum;
         }
      }
      return;
   }

   static void pressureBackstreamCalculations( const SpatialCell * cell, 
                                               const bool calculateBackstream, 
                                               const Real averageVX,
                                               const Real averageVY,
                                               const Real averageVZ,
                                               Real & Pressure ) {
      const Real HALF = 0.5;
      const Real THIRD = 1.0/3.0;
      Pressure = 0;
      # pragma omp parallel
      {
         Real thread_nvx2_sum = 0.0;
         Real thread_nvy2_sum = 0.0;
         Real thread_nvz2_sum = 0.0;
         # pragma omp for
         for (vmesh::LocalID n=0; n<cell->get_number_of_velocity_blocks(); ++n) {
            const Real* parameters = cell->get_block_parameters(n);
            const Realf* block_data = cell->get_data(n);
            const Real DV3 = parameters[BlockParams::DVX] * parameters[BlockParams::DVY] * parameters[BlockParams::DVZ];
            vector< array<uint, 3> > vCellIndices;
            vCellIndices.clear();
            //Note: Could use function pointers
            if( calculateBackstream == true ) {
               getBackstreamVelocityCellIndices(parameters, vCellIndices);
            } else {
               getNonBackstreamVelocityCellIndices(parameters, vCellIndices);
            }
            for( vector< array<uint, 3> >::const_iterator it = vCellIndices.begin(); it != vCellIndices.end(); ++it ) {
               //Go through every velocity cell:
               const array<uint, 3> indices = *it;
               const uint i = indices[0];
               const uint j = indices[1];
               const uint k = indices[2];
               const Real VX = parameters[BlockParams::VXCRD] + (i+HALF) * parameters[BlockParams::DVX];
               const Real VY = parameters[BlockParams::VYCRD] + (j+HALF) * parameters[BlockParams::DVY];
               const Real VZ = parameters[BlockParams::VZCRD] + (k+HALF) * parameters[BlockParams::DVZ];
               thread_nvx2_sum += block_data[cellIndex(i,j,k)] * (VX - averageVX) * (VX - averageVX) * DV3;
               thread_nvy2_sum += block_data[cellIndex(i,j,k)] * (VY - averageVY) * (VY - averageVY) * DV3;
               thread_nvz2_sum += block_data[cellIndex(i,j,k)] * (VZ - averageVZ) * (VZ - averageVZ) * DV3;
            }
         }
         
         // Accumulate contributions coming from this velocity block to the 
         // spatial cell velocity moments. If multithreading / OpenMP is used, 
         // these updates need to be atomic:
         # pragma omp critical
         {
            Pressure += physicalconstants::MASS_PROTON * THIRD * (thread_nvx2_sum + thread_nvy2_sum + thread_nvz2_sum);
         }
      }
   }

   static void PTensorDiagonalBackstreamCalculations( const SpatialCell * cell,
                                                      const bool calculateBackstream,
                                                      const Real averageVX,
                                                      const Real averageVY,
                                                      const Real averageVZ,
                                                      Real * PTensor ) {
      const Real HALF = 0.5;
      # pragma omp parallel
      {
         Real thread_nvxvx_sum = 0.0;
         Real thread_nvyvy_sum = 0.0;
         Real thread_nvzvz_sum = 0.0;
         # pragma omp for
         for (vmesh::LocalID n=0; n<cell->get_number_of_velocity_blocks(); ++n) {
            const Real* parameters = cell->get_block_parameters(n);
            const Realf* block_data = cell->get_data(n);
            const Real DV3 = parameters[BlockParams::DVX] * parameters[BlockParams::DVY] * parameters[BlockParams::DVZ];
            vector< array<uint, 3> > vCellIndices;
            vCellIndices.clear();
            if( calculateBackstream == true ) {
               getBackstreamVelocityCellIndices(parameters, vCellIndices);
            } else {
               getNonBackstreamVelocityCellIndices(parameters, vCellIndices);
            }
            for( vector< array<uint, 3> >::const_iterator it = vCellIndices.begin(); it != vCellIndices.end(); ++it ) {
               //Go through every velocity cell:
               const array<uint, 3> indices = *it;
               const uint i = indices[0];
               const uint j = indices[1];
               const uint k = indices[2];
               const Real VX = parameters[BlockParams::VXCRD] + (i+HALF) * parameters[BlockParams::DVX];
               const Real VY = parameters[BlockParams::VYCRD] + (j+HALF) * parameters[BlockParams::DVY];
               const Real VZ = parameters[BlockParams::VZCRD] + (k+HALF) * parameters[BlockParams::DVZ];
               thread_nvxvx_sum += block_data[cellIndex(i,j,k)] * (VX - averageVX) * (VX - averageVX) * DV3;
               thread_nvyvy_sum += block_data[cellIndex(i,j,k)] * (VY - averageVY) * (VY - averageVY) * DV3;
               thread_nvzvz_sum += block_data[cellIndex(i,j,k)] * (VZ - averageVZ) * (VZ - averageVZ) * DV3;
            }
         }
         
         // Accumulate contributions coming from this velocity block to the 
         // spatial cell velocity moments. If multithreading / OpenMP is used, 
         // these updates need to be atomic:
         # pragma omp critical
         {
            PTensor[0] += physicalconstants::MASS_PROTON * thread_nvxvx_sum;
            PTensor[1] += physicalconstants::MASS_PROTON * thread_nvyvy_sum;
            PTensor[2] += physicalconstants::MASS_PROTON * thread_nvzvz_sum;
         }
      }
      return;
   }

   static void PTensorOffDiagonalBackstreamCalculations( const SpatialCell * cell,
                                                         const bool calculateBackstream,
                                                         const Real averageVX,
                                                         const Real averageVY,
                                                         const Real averageVZ,
                                                         Real * PTensor ) {
      const Real HALF = 0.5;
      # pragma omp parallel
      {
         Real thread_nvxvy_sum = 0.0;
         Real thread_nvzvx_sum = 0.0;
         Real thread_nvyvz_sum = 0.0;
         # pragma omp for
         for (vmesh::LocalID n=0; n<cell->get_number_of_velocity_blocks(); ++n) {
            const Real* parameters = cell->get_block_parameters(n);
            const Realf* block_data = cell->get_data(n);
            const Real DV3 = parameters[BlockParams::DVX] * parameters[BlockParams::DVY] * parameters[BlockParams::DVZ];
            vector< array<uint, 3> > vCellIndices;
            if( calculateBackstream == true ) {
               getBackstreamVelocityCellIndices(parameters, vCellIndices);
            } else {
               getNonBackstreamVelocityCellIndices(parameters, vCellIndices);
            }
            for( vector< array<uint, 3> >::const_iterator it = vCellIndices.begin(); it != vCellIndices.end(); ++it ) {
               //Go through every velocity cell:
               const array<uint, 3> indices = *it;
               const uint i = indices[0];
               const uint j = indices[1];
               const uint k = indices[2];
               const Real VX = parameters[BlockParams::VXCRD] + (i+HALF) * parameters[BlockParams::DVX];
               const Real VY = parameters[BlockParams::VYCRD] + (j+HALF) * parameters[BlockParams::DVY];
               const Real VZ = parameters[BlockParams::VZCRD] + (k+HALF) * parameters[BlockParams::DVZ];
               thread_nvxvy_sum += block_data[cellIndex(i,j,k)] * (VX - averageVX) * (VY - averageVY) * DV3;
               thread_nvzvx_sum += block_data[cellIndex(i,j,k)] * (VZ - averageVZ) * (VX - averageVX) * DV3;
               thread_nvyvz_sum += block_data[cellIndex(i,j,k)] * (VY - averageVY) * (VZ - averageVZ) * DV3;
            }
         }
         
         // Accumulate contributions coming from this velocity block to the 
         // spatial cell velocity moments. If multithreading / OpenMP is used, 
         // these updates need to be atomic:
         # pragma omp critical
         {
            PTensor[0] += physicalconstants::MASS_PROTON * thread_nvyvz_sum;
            PTensor[1] += physicalconstants::MASS_PROTON * thread_nvzvx_sum;
            PTensor[2] += physicalconstants::MASS_PROTON * thread_nvxvy_sum;
         }
      }
   }


   // Rho backstream:
   VariableRhoBackstream::VariableRhoBackstream(): DataReductionOperator() { }
   VariableRhoBackstream::~VariableRhoBackstream() { }
   
   std::string VariableRhoBackstream::getName() const {return "RhoBackstream";}
   
   bool VariableRhoBackstream::getDataVectorInfo(std::string& dataType,unsigned int& dataSize,unsigned int& vectorSize) const {
      dataType = "float";
      dataSize =  sizeof(Real);
      vectorSize = 1;
      return true;
   }
   
   // Adding rho backstream calculations to Vlasiator.
   bool VariableRhoBackstream::reduceData(const SpatialCell* cell,char* buffer) {
      const bool calculateBackstream = true;
      rhoBackstreamCalculation( cell, calculateBackstream, RhoBackstream );
      const char* ptr = reinterpret_cast<const char*>(&RhoBackstream);
      for (uint i=0; i<sizeof(Real); ++i) buffer[i] = ptr[i];
      return true;
   }
   
   bool VariableRhoBackstream::setSpatialCell(const SpatialCell* cell) {
      RhoBackstream = 0.0;
      return true;
   }


   // Rho non backstream:
   VariableRhoNonBackstream::VariableRhoNonBackstream(): DataReductionOperator() { }
   VariableRhoNonBackstream::~VariableRhoNonBackstream() { }
   
   std::string VariableRhoNonBackstream::getName() const {return "RhoNonBackstream";}
   
   bool VariableRhoNonBackstream::getDataVectorInfo(std::string& dataType,unsigned int& dataSize,unsigned int& vectorSize) const {
      dataType = "float";
      dataSize =  sizeof(Real);
      vectorSize = 1;
      return true;
   }
   
   // Adding rho non backstream calculations to Vlasiator.
   bool VariableRhoNonBackstream::reduceData(const SpatialCell* cell,char* buffer) {
      const bool calculateBackstream = false; //We don't want backstream
      rhoBackstreamCalculation( cell, calculateBackstream, Rho );
      const char* ptr = reinterpret_cast<const char*>(&Rho);
      for (uint i=0; i<sizeof(Real); ++i) buffer[i] = ptr[i];
      return true;
   }
   
   bool VariableRhoNonBackstream::setSpatialCell(const SpatialCell* cell) {
      Rho = 0.0;
      return true;
   }

   //Rho v backstream:
   VariableRhoVBackstream::VariableRhoVBackstream(): DataReductionOperator() { }
   VariableRhoVBackstream::~VariableRhoVBackstream() { }
   
   std::string VariableRhoVBackstream::getName() const {return "RhoVBackstream";}
   
   bool VariableRhoVBackstream::getDataVectorInfo(std::string& dataType,unsigned int& dataSize,unsigned int& vectorSize) const {
      dataType = "float";
      dataSize =  sizeof(Real);
      vectorSize = 3;
      return true;
   }



   // Adding rho v backstream calculations to Vlasiator.
   bool VariableRhoVBackstream::reduceData(const SpatialCell* cell,char* buffer) {
      const bool calculateBackstream = true;
      //Calculate rho v backstream
      rhoVBackstreamCalculation( cell, calculateBackstream, RhoVBackstream );
      const uint RhoVBackstreamSize = 3;
      const char* ptr = reinterpret_cast<const char*>(&RhoVBackstream);
      for (uint i=0; i<RhoVBackstreamSize*sizeof(Real); ++i) buffer[i] = ptr[i];
      return true;
   }
   
   bool VariableRhoVBackstream::setSpatialCell(const SpatialCell* cell) {
      // Initialize values
      for( uint i = 0; i < 3; ++i ) {
         RhoVBackstream[i] = 0.0;
      }
      return true;
   }

   //Rho v non backstream:
   VariableRhoVNonBackstream::VariableRhoVNonBackstream(): DataReductionOperator() { }
   VariableRhoVNonBackstream::~VariableRhoVNonBackstream() { }
   
   std::string VariableRhoVNonBackstream::getName() const {return "RhoVNonBackstream";}
   
   bool VariableRhoVNonBackstream::getDataVectorInfo(std::string& dataType,unsigned int& dataSize,unsigned int& vectorSize) const {
      dataType = "float";
      dataSize =  sizeof(Real);
      vectorSize = 3;
      return true;
   }

   // Adding rho v non backstream calculations to Vlasiator.
   bool VariableRhoVNonBackstream::reduceData(const SpatialCell* cell,char* buffer) {
      const bool calculateBackstream = false;
      //Calculate rho v backstream
      rhoVBackstreamCalculation( cell, calculateBackstream, RhoV );
      const uint vectorSize = 3;
      const char* ptr = reinterpret_cast<const char*>(&RhoV);
      for (uint i=0; i<vectorSize*sizeof(Real); ++i) buffer[i] = ptr[i];
      return true;
   }
   
   bool VariableRhoVNonBackstream::setSpatialCell(const SpatialCell* cell) {
      // Initialize values
      for( uint i = 0; i < 3; ++i ) {
         RhoV[i] = 0.0;
      }
      return true;
   }


   // Scalar pressure of backstream
   VariablePressureBackstream::VariablePressureBackstream(): DataReductionOperator() { }
   VariablePressureBackstream::~VariablePressureBackstream() { }
   
   std::string VariablePressureBackstream::getName() const {return "PressureBackstream";}
   
   bool VariablePressureBackstream::getDataVectorInfo(std::string& dataType,unsigned int& dataSize,unsigned int& vectorSize) const {
      dataType = "float";
      dataSize =  sizeof(Real);
      vectorSize = 1;
      return true;
   }
   
   // Adding pressure backstream calculations to Vlasiator.
   // p = m/3 * integral((v - <V>)^2 * f(r,v) dV), doing the sum of the x, y and z components.
   bool VariablePressureBackstream::reduceData(const SpatialCell* cell,char* buffer) {
      const bool calculateBackstream = true;
      pressureBackstreamCalculations( cell, calculateBackstream, averageVX, averageVY, averageVZ, Pressure );
      const char* ptr = reinterpret_cast<const char*>(&Pressure);
      for (uint i=0; i<sizeof(Real); ++i) buffer[i] = ptr[i];
      return true;
   }
   
   bool VariablePressureBackstream::setSpatialCell(const SpatialCell* cell) {
      if(cell-> parameters[CellParams::RHO] != 0.0) {
         //Get rho and rho v of the backstream:
         Real rho = 0;
         Real rhoV[3] = {0};
         const bool calculateBackstream = true;
         rhoBackstreamCalculation( cell, calculateBackstream, rho );
         rhoVBackstreamCalculation( cell, calculateBackstream, rhoV );
         //Set the average velocities:
         averageVX = rhoV[0] / rho;
         averageVY = rhoV[1] / rho;
         averageVZ = rhoV[2] / rho;
      } else {
         averageVX = 0.0;
         averageVY = 0.0;
         averageVZ = 0.0;
      }
      Pressure = 0.0;
      return true;
   }

   // Scalar pressure of non backstream
   VariablePressureNonBackstream::VariablePressureNonBackstream(): DataReductionOperator() { }
   VariablePressureNonBackstream::~VariablePressureNonBackstream() { }
   
   std::string VariablePressureNonBackstream::getName() const {return "PressureNonBackstream";}
   
   bool VariablePressureNonBackstream::getDataVectorInfo(std::string& dataType,unsigned int& dataSize,unsigned int& vectorSize) const {
      dataType = "float";
      dataSize =  sizeof(Real);
      vectorSize = 1;
      return true;
   }
   
   // Adding pressure backstream calculations to Vlasiator.
   // p = m/3 * integral((v - <V>)^2 * f(r,v) dV), doing the sum of the x, y and z components.
   bool VariablePressureNonBackstream::reduceData(const SpatialCell* cell,char* buffer) {
      const bool calculateBackstream = false;
      pressureBackstreamCalculations( cell, calculateBackstream, averageVX, averageVY, averageVZ, Pressure );
      const char* ptr = reinterpret_cast<const char*>(&Pressure);
      for (uint i=0; i<sizeof(Real); ++i) buffer[i] = ptr[i];
      return true;
   }
   
   bool VariablePressureNonBackstream::setSpatialCell(const SpatialCell* cell) {
      if(cell-> parameters[CellParams::RHO] != 0.0) {
         //Get rho and rho v of the backstream:
         Real rho = 0;
         Real rhoV[3] = {0};
         const bool calculateBackstream = false;
         rhoBackstreamCalculation( cell, calculateBackstream, rho );
         rhoVBackstreamCalculation( cell, calculateBackstream, rhoV );
         //Set the average velocities:
         averageVX = rhoV[0] / rho;
         averageVY = rhoV[1] / rho;
         averageVZ = rhoV[2] / rho;
      } else {
         averageVX = 0.0;
         averageVY = 0.0;
         averageVZ = 0.0;
      }
      Pressure = 0.0;
      return true;
   }


   // Adding pressure calculations for backstream population to Vlasiator.
   // p_ij = m/3 * integral((v - <V>)_i(v - <V>)_j * f(r,v) dV)
   
   // Pressure tensor 6 components (11, 22, 33, 23, 13, 12) added by YK
   // Split into VariablePTensorBackstreamDiagonal (11, 22, 33)
   // and VariablePTensorOffDiagonal (23, 13, 12)
   VariablePTensorBackstreamDiagonal::VariablePTensorBackstreamDiagonal(): DataReductionOperator() { }
   VariablePTensorBackstreamDiagonal::~VariablePTensorBackstreamDiagonal() { }
   
   std::string VariablePTensorBackstreamDiagonal::getName() const {return "PTensorBackstreamDiagonal";}
   
   bool VariablePTensorBackstreamDiagonal::getDataVectorInfo(std::string& dataType,unsigned int& dataSize,unsigned int& vectorSize) const {
      dataType = "float";
      dataSize =  sizeof(Real);
      vectorSize = 3;
      return true;
   }
   
   bool VariablePTensorBackstreamDiagonal::reduceData(const SpatialCell* cell,char* buffer) {
      const bool calculateBackstream = true;
      //Calculate PTensor and save it in PTensorArray:
      PTensorDiagonalBackstreamCalculations( cell, calculateBackstream, averageVX, averageVY, averageVZ, PTensor );
      const uint vectorSize = 3;
      //Save the data into buffer:
      const char* ptr = reinterpret_cast<const char*>(&PTensor);
      for (uint i=0; i<vectorSize*sizeof(Real); ++i) buffer[i] = ptr[i];
      return true;
   }
   
   bool VariablePTensorBackstreamDiagonal::setSpatialCell(const SpatialCell* cell) {
      if(cell-> parameters[CellParams::RHO] != 0.0) {
         //Get rho and rho v of the backstream:
         Real rho = 0;
         Real rhoV[3] = {0};
         const bool calculateBackstream = true; //We are calculating backstream
         rhoBackstreamCalculation( cell, calculateBackstream, rho );
         rhoVBackstreamCalculation( cell, calculateBackstream, rhoV );
         //Set the average velocities:
         averageVX = rhoV[0] / rho;
         averageVY = rhoV[1] / rho;
         averageVZ = rhoV[2] / rho;
      } else {
         averageVX = 0.0;
         averageVY = 0.0;
         averageVZ = 0.0;
      }
      const uint vectorSize = 3;
      for(uint i = 0; i < vectorSize; i++) PTensor[i] = 0.0;
      return true;
   }

   // Adding pressure calculations for backstream population to Vlasiator.
   // p_ij = m/3 * integral((v - <V>)_i(v - <V>)_j * f(r,v) dV)
   
   // Pressure tensor 6 components (11, 22, 33, 23, 13, 12) added by YK
   // Split into VariablePTensorNonBackstreamDiagonal (11, 22, 33)
   // and VariablePTensorOffDiagonal (23, 13, 12)
   VariablePTensorNonBackstreamDiagonal::VariablePTensorNonBackstreamDiagonal(): DataReductionOperator() { }
   VariablePTensorNonBackstreamDiagonal::~VariablePTensorNonBackstreamDiagonal() { }
   
   std::string VariablePTensorNonBackstreamDiagonal::getName() const {return "PTensorNonBackstreamDiagonal";}
   
   bool VariablePTensorNonBackstreamDiagonal::getDataVectorInfo(std::string& dataType,unsigned int& dataSize,unsigned int& vectorSize) const {
      dataType = "float";
      dataSize =  sizeof(Real);
      vectorSize = 3;
      return true;
   }
   
   bool VariablePTensorNonBackstreamDiagonal::reduceData(const SpatialCell* cell,char* buffer) {
      const bool calculateBackstream = false;
      //Calculate PTensor and save it in PTensorArray:
      PTensorDiagonalBackstreamCalculations( cell, calculateBackstream, averageVX, averageVY, averageVZ, PTensor );
      const uint vectorSize = 3;
      //Save the data into buffer:
      const char* ptr = reinterpret_cast<const char*>(&PTensor);
      for (uint i=0; i<vectorSize*sizeof(Real); ++i) buffer[i] = ptr[i];
      return true;
   }
   
   bool VariablePTensorNonBackstreamDiagonal::setSpatialCell(const SpatialCell* cell) {
      if(cell-> parameters[CellParams::RHO] != 0.0) {
         //Get rho and rho v of the backstream:
         Real rho = 0;
         Real rhoV[3] = {0};
         const bool calculateBackstream = false; //We are not calculating backstream
         rhoBackstreamCalculation( cell, calculateBackstream, rho );
         rhoVBackstreamCalculation( cell, calculateBackstream, rhoV );
         //Set the average velocities:
         averageVX = rhoV[0] / rho;
         averageVY = rhoV[1] / rho;
         averageVZ = rhoV[2] / rho;
      } else {
         averageVX = 0.0;
         averageVY = 0.0;
         averageVZ = 0.0;
      }
      const uint vectorSize = 3;
      for(uint i = 0; i < vectorSize; i++) PTensor[i] = 0.0;
      return true;
   }

   VariablePTensorBackstreamOffDiagonal::VariablePTensorBackstreamOffDiagonal(): DataReductionOperator() { }
   VariablePTensorBackstreamOffDiagonal::~VariablePTensorBackstreamOffDiagonal() { }
   
   std::string VariablePTensorBackstreamOffDiagonal::getName() const {return "PTensorBackstreamOffDiagonal";}
   
   bool VariablePTensorBackstreamOffDiagonal::getDataVectorInfo(std::string& dataType,unsigned int& dataSize,unsigned int& vectorSize) const {
      dataType = "float";
      dataSize =  sizeof(Real);
      vectorSize = 3;
      return true;
   }
   
   bool VariablePTensorBackstreamOffDiagonal::reduceData(const SpatialCell* cell,char* buffer) {
      //Calculate PTensor for PTensorArray:
      const bool calculateBackstream = true;
      //Calculate and save:
      PTensorOffDiagonalBackstreamCalculations( cell, calculateBackstream, averageVX, averageVY, averageVZ, PTensor );
      const uint vectorSize = 3;
      //Input data into buffer
      const char* ptr = reinterpret_cast<const char*>(&PTensor);
      for (uint i=0; i<3*sizeof(Real); ++i) buffer[i] = ptr[i];
      return true;
   }
   
   bool VariablePTensorBackstreamOffDiagonal::setSpatialCell(const SpatialCell* cell) {
      if(cell-> parameters[CellParams::RHO] != 0.0) {
         //Get rho and rho v of the backstream:
         Real rho = 0;
         Real rhoV[3] = {0};
         const bool calculateBackstream = true; //We are calculating backstream
         rhoBackstreamCalculation( cell, calculateBackstream, rho );
         rhoVBackstreamCalculation( cell, calculateBackstream, rhoV );
         //Set the average velocities:
         averageVX = rhoV[0] / rho;
         averageVY = rhoV[1] / rho;
         averageVZ = rhoV[2] / rho;
      } else {
         averageVX = 0.0;
         averageVY = 0.0;
         averageVZ = 0.0;
      }
      for(int i = 0; i < 3; i++) PTensor[i] = 0.0;
      return true;
   }

   VariablePTensorNonBackstreamOffDiagonal::VariablePTensorNonBackstreamOffDiagonal(): DataReductionOperator() { }
   VariablePTensorNonBackstreamOffDiagonal::~VariablePTensorNonBackstreamOffDiagonal() { }
   
   std::string VariablePTensorNonBackstreamOffDiagonal::getName() const {return "PTensorNonBackstreamOffDiagonal";}
   
   bool VariablePTensorNonBackstreamOffDiagonal::getDataVectorInfo(std::string& dataType,unsigned int& dataSize,unsigned int& vectorSize) const {
      dataType = "float";
      dataSize =  sizeof(Real);
      vectorSize = 3;
      return true;
   }
   
   bool VariablePTensorNonBackstreamOffDiagonal::reduceData(const SpatialCell* cell,char* buffer) {
      //Calculate PTensor for PTensorArray:
      const bool calculateBackstream = false;
      //Calculate and save:
      PTensorOffDiagonalBackstreamCalculations( cell, calculateBackstream, averageVX, averageVY, averageVZ, PTensor );
      const uint vectorSize = 3;
      //Input data into buffer
      const char* ptr = reinterpret_cast<const char*>(&PTensor);
      for (uint i=0; i<3*sizeof(Real); ++i) buffer[i] = ptr[i];
      return true;
   }
   
   bool VariablePTensorNonBackstreamOffDiagonal::setSpatialCell(const SpatialCell* cell) {
      if(cell-> parameters[CellParams::RHO] != 0.0) {
         //Get rho and rho v of the backstream:
         Real rho = 0;
         Real rhoV[3] = {0};
         const bool calculateBackstream = false; //We are not calculating backstream
         rhoBackstreamCalculation( cell, calculateBackstream, rho );
         rhoVBackstreamCalculation( cell, calculateBackstream, rhoV );
         //Set the average velocities:
         averageVX = rhoV[0] / rho;
         averageVY = rhoV[1] / rho;
         averageVZ = rhoV[2] / rho;
      } else {
         averageVX = 0.0;
         averageVY = 0.0;
         averageVZ = 0.0;
      }
      for(int i = 0; i < 3; i++) PTensor[i] = 0.0;
      return true;
   }

   // Adding pressure calculations for backstream population to Vlasiator.
   // p_ij = m/3 * integral((v - <V>)_i(v - <V>)_j * f(r,v) dV)
   
   // Pressure tensor 6 Xcomponents (11, 22, 33, 23, 13, 12) added by YK
   // Split into VariablePTensorBackstreamDiagonal (11, 22, 33)
   // and VariablePTensorOffDiagonal (23, 13, 12)
   VariableMinValue::VariableMinValue(): DataReductionOperator() { }
   VariableMinValue::~VariableMinValue() { }
   
   std::string VariableMinValue::getName() const {return "MinValue";}
   
   bool VariableMinValue::getDataVectorInfo(std::string& dataType,unsigned int& dataSize,unsigned int& vectorSize) const {
      dataType = "float";
      dataSize =  sizeof(Real);
      vectorSize = 1;
      return true;
   }
   
   bool VariableMinValue::reduceData(const SpatialCell* cell,char* buffer) {
      const uint vectorSize = 1;
      const Real minValue = cell->getVelocityBlockMinValue();
      //Save the data into buffer:
      const char* ptr = reinterpret_cast<const char*>(&minValue);
      for (uint i=0; i<vectorSize*sizeof(Real); ++i) buffer[i] = ptr[i];
      return true;
   }
   
   bool VariableMinValue::setSpatialCell(const SpatialCell* cell) {
      return true;
   }
   
   

   VariableNumberOfPopulations::VariableNumberOfPopulations(): DataReductionOperator() { }
   VariableNumberOfPopulations::~VariableNumberOfPopulations() { }
   
   std::string VariableNumberOfPopulations::getName() const {return "NumberOfPopulations";}
   
   bool VariableNumberOfPopulations::getDataVectorInfo(std::string& dataType,unsigned int& dataSize,unsigned int& vectorSize) const {
      dataType = "uint";
      dataSize =  sizeof(uint);
      vectorSize = 1;
      return true;
   }
   
   // Adding rho non backstream calculations to Vlasiator.
   bool VariableNumberOfPopulations::reduceData(const SpatialCell* cell,char* buffer) {
      const char* ptr = reinterpret_cast<const char*>(&cell->number_of_populations);
      for (uint i=0; i<sizeof(cell->number_of_populations); ++i) buffer[i] = ptr[i];
      return true;
   }
   
   bool VariableNumberOfPopulations::setSpatialCell(const SpatialCell* cell) {
      return true;
   }

   VariablePopulationRho::VariablePopulationRho(): DataReductionOperator() { }
   VariablePopulationRho::~VariablePopulationRho() { }
   
   std::string VariablePopulationRho::getName( const int population ) const {
     stringstream ss(stringstream::in | stringstream::out );
     ss << "populations/";
     ss << population;
     ss << "/rho";
     return ss.str();
   }
   
   bool VariablePopulationRho::getDataVectorInfo(std::string& dataType,unsigned int& dataSize,unsigned int& vectorSize) const {
      dataType = "float";
      dataSize =  sizeof(Real);
      vectorSize = 1;
      return true;
   }

   // Adding rho non backstream calculations to Vlasiator.
   bool VariablePopulationRho::reduceData(const SpatialCell* cell, const int population, char* buffer) {
      const char* ptr = reinterpret_cast<const char*>(&cell->populationParameters[PopulationParams::POPULATION_RHO][population]);
      if( PopulationParams::POPULATION_RHO >= PopulationParams::N_POPULATION_PARAMS ) {
         cerr << "BAD POP PARAMS " << PopulationParams::POPULATION_RHO << " " << PopulationParams::N_POPULATION_PARAMS << endl;
      }
      if( population >= cell->populationParameters[PopulationParams::POPULATION_RHO].size() ) { 
         cerr << "BAD POPULATION: " << cell->populationParameters[PopulationParams::POPULATION_RHO].size() << " " << population << " " << cell->number_of_populations << " " << cell->get_number_of_velocity_blocks() << endl; 
      }
      for (uint i=0; i<sizeof(cell->populationParameters[PopulationParams::POPULATION_RHO][population]); ++i) {
         buffer[i] = ptr[i];
      }
      return true;
   }

   bool VariablePopulationRho::setSpatialCell(const SpatialCell* cell) {
      return true;
   }

  
} // namespace DRO
