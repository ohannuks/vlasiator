/*
This file is part of Vlasiator.

Copyright 2010, 2011, 2012, 2013 Finnish Meteorological Institute












*/

#include <cstdlib>
#include <iostream>

#include "datareducer.h"
#include "../common.h"
using namespace std;

void initializeDataReducers(DataReducer * outputReducer, DataReducer * diagnosticReducer, DataReducer * populationReducer)
{
   typedef Parameters P;
   /*
     //TODO - make these optional in cfg
     outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("X",CellParams::XCRD,1));
     outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("Y",CellParams::YCRD,1));
     outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("Z",CellParams::ZCRD,1));
     outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("DX",CellParams::DX,1));
     outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("DY",CellParams::DY,1));
     outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("DZ",CellParams::DZ,1));
   */

   vector<string>::const_iterator it;
   for (it = P::outputVariableList.begin();
        it != P::outputVariableList.end();
        it++) {
      if(*it == "B")
         outputReducer->addOperator(new DRO::VariableB);
      if(*it == "BackgroundB")
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("background_B",CellParams::BGBX,3));
      if(*it == "PerturbedB")
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("perturbed_B",CellParams::PERBX,3));
      if(*it == "E")
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("E",CellParams::EX,3));
      if(*it == "Rho")
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("rho",CellParams::RHO,1));
      if(*it == "RhoBackstream")
         outputReducer->addOperator(new DRO::VariableRhoBackstream);
      if(*it == "RhoV")
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("rho_v",CellParams::RHOVX,3));
      if(*it == "RhoVBackstream")
         outputReducer->addOperator(new DRO::VariableRhoVBackstream);
      if(*it == "RhoVNonBackstream")
         outputReducer->addOperator(new DRO::VariableRhoVNonBackstream);
      if(*it == "PressureBackstream")
         outputReducer->addOperator(new DRO::VariablePressureBackstream);
      if(*it == "PTensorBackstreamDiagonal")
         outputReducer->addOperator(new DRO::VariablePTensorBackstreamDiagonal);
      if(*it == "PTensorNonBackstreamDiagonal")
         outputReducer->addOperator(new DRO::VariablePTensorNonBackstreamDiagonal);
      if(*it == "PTensorBackstreamOffDiagonal")
         outputReducer->addOperator(new DRO::VariablePTensorBackstreamOffDiagonal);
      if(*it == "PTensorNonBackstreamOffDiagonal")
         outputReducer->addOperator(new DRO::VariablePTensorNonBackstreamOffDiagonal);
      if(*it == "PTensorBackstream") {
         outputReducer->addOperator(new DRO::VariablePTensorBackstreamDiagonal);
         outputReducer->addOperator(new DRO::VariablePTensorBackstreamOffDiagonal);
      }
      if(*it == "PTensorNonBackstream") {
         outputReducer->addOperator(new DRO::VariablePTensorNonBackstreamDiagonal);
         outputReducer->addOperator(new DRO::VariablePTensorNonBackstreamOffDiagonal);
      }
      if(*it == "MinValue") {
         outputReducer->addOperator(new DRO::VariableMinValue);
      }
      if(*it == "RhoNonBackstream")
         outputReducer->addOperator(new DRO::VariableRhoNonBackstream);
      if(*it == "RhoLossAdjust")
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("rho_loss_adjust",CellParams::RHOLOSSADJUST,1));
      if(*it == "RhoLossVelBoundary")
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("rho_loss_velocity_boundary",CellParams::RHOLOSSVELBOUNDARY,1));
      if(*it == "LBweight")
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("LB_weight",CellParams::LBWEIGHTCOUNTER,1));
      if(*it == "MaxVdt")
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("max_v_dt",CellParams::MAXVDT,1));
      if(*it == "MaxRdt")
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("max_r_dt",CellParams::MAXRDT,1));
      if(*it == "MaxFieldsdt")
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("max_fields_dt",CellParams::MAXFDT,1));
      if(*it == "MPIrank")
         outputReducer->addOperator(new DRO::MPIrank);
      if(*it == "BoundaryType")
         outputReducer->addOperator(new DRO::BoundaryType);
      if(*it == "BoundaryLayer")
         outputReducer->addOperator(new DRO::BoundaryLayer);
      if(*it == "Blocks")
         outputReducer->addOperator(new DRO::Blocks);
      if(*it == "fSaved")
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("fSaved",CellParams::ISCELLSAVINGF,1));
      if(*it == "accSubcycles")
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("acc_subcycles",CellParams::ACCSUBCYCLES,1));
      if(*it == "VolE")
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("E_vol",CellParams::EXVOL,3));
      if(*it == "NumberOfPopulations")
         outputReducer->addOperator(new DRO::VariableNumberOfPopulations);
      if(*it == "HallE") {
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("EXHALL_000_100",CellParams::EXHALL_000_100,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("EXHALL_001_101",CellParams::EXHALL_001_101,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("EXHALL_010_110",CellParams::EXHALL_010_110,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("EXHALL_011_111",CellParams::EXHALL_011_111,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("EYHALL_000_010",CellParams::EYHALL_000_010,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("EYHALL_001_011",CellParams::EYHALL_001_011,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("EYHALL_100_110",CellParams::EYHALL_100_110,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("EYHALL_101_111",CellParams::EYHALL_101_111,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("EZHALL_000_001",CellParams::EZHALL_000_001,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("EZHALL_010_011",CellParams::EZHALL_010_011,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("EZHALL_100_101",CellParams::EZHALL_100_101,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("EZHALL_110_111",CellParams::EZHALL_110_111,1));
      }
      if(*it == "BackgroundBedge") {
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("BGBX_000_010",CellParams::BGBX_000_010,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("BGBX_100_110",CellParams::BGBX_100_110,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("BGBX_001_011",CellParams::BGBX_001_011,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("BGBX_101_111",CellParams::BGBX_101_111,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("BGBX_000_001",CellParams::BGBX_000_001,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("BGBX_100_101",CellParams::BGBX_100_101,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("BGBX_010_011",CellParams::BGBX_010_011,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("BGBX_110_111",CellParams::BGBX_110_111,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("BGBY_000_100",CellParams::BGBY_000_100,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("BGBY_010_110",CellParams::BGBY_010_110,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("BGBY_001_101",CellParams::BGBY_001_101,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("BGBY_011_111",CellParams::BGBY_011_111,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("BGBY_000_001",CellParams::BGBY_000_001,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("BGBY_100_101",CellParams::BGBY_100_101,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("BGBY_010_011",CellParams::BGBY_010_011,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("BGBY_110_111",CellParams::BGBY_110_111,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("BGBZ_000_100",CellParams::BGBZ_000_100,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("BGBZ_010_110",CellParams::BGBZ_010_110,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("BGBZ_001_101",CellParams::BGBZ_001_101,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("BGBZ_011_111",CellParams::BGBZ_011_111,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("BGBZ_000_010",CellParams::BGBZ_000_010,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("BGBZ_100_110",CellParams::BGBZ_100_110,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("BGBZ_001_011",CellParams::BGBZ_001_011,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("BGBZ_101_111",CellParams::BGBZ_101_111,1));
         
      }
      if(*it == "VolB")
         outputReducer->addOperator(new DRO::VariableBVol);
      if(*it == "BackgroundVolB")
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("BGB_vol",CellParams::BGBXVOL,3));
      if(*it == "PerturbedVolB")
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("PERB_vol",CellParams::PERBXVOL,3));
      if(*it == "Pressure") {
         outputReducer->addOperator(new DRO::VariablePressureSolver);
      }
      if(*it == "PTensor") {
         outputReducer->addOperator(new DRO::VariablePTensorDiagonal);
         outputReducer->addOperator(new DRO::VariablePTensorOffDiagonal);
      }
      if(*it == "derivs") {
         outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("drhodx",fieldsolver::drhodx,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("drhody",fieldsolver::drhody,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("drhodz",fieldsolver::drhodz,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("dp11dx",fieldsolver::dp11dx,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("dp22dx",fieldsolver::dp22dx,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("dp33dx",fieldsolver::dp33dx,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("dp11dy",fieldsolver::dp11dy,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("dp22dy",fieldsolver::dp22dy,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("dp33dy",fieldsolver::dp33dy,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("dp11dz",fieldsolver::dp11dz,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("dp22dz",fieldsolver::dp22dz,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("dp33dz",fieldsolver::dp33dz,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("dPERBxdy",fieldsolver::dPERBxdy,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("dBGBxdy",fieldsolver::dBGBxdy,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("dPERBxdz",fieldsolver::dPERBxdz,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("dBGBxdz",fieldsolver::dBGBxdz,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("dPERBydx",fieldsolver::dPERBydx,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("dBGBydx",fieldsolver::dBGBydx,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("dPERBydz",fieldsolver::dPERBydz,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("dBGBydz",fieldsolver::dBGBydz,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("dPERBzdx",fieldsolver::dPERBzdx,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("dBGBzdx",fieldsolver::dBGBzdx,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("dPERBzdy",fieldsolver::dPERBzdy,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("dBGBzdy",fieldsolver::dBGBzdy,1));
         if(Parameters::ohmHallTerm == 2) {
            outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("dPERBxdyy",fieldsolver::dPERBxdyy,1));
            outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("dPERBxdzz",fieldsolver::dPERBxdzz,1));
            outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("dPERBydxx",fieldsolver::dPERBydxx,1));
            outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("dPERBydzz",fieldsolver::dPERBydzz,1));
            outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("dPERBzdxx",fieldsolver::dPERBzdxx,1));
            outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("dPERBzdyy",fieldsolver::dPERBzdyy,1));
            outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("dPERBxdyz",fieldsolver::dPERBxdyz,1));
            outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("dPERBydxz",fieldsolver::dPERBydxz,1));
            outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("dPERBzdxy",fieldsolver::dPERBzdxy,1));
         }
         outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("dVxdx",fieldsolver::dVxdx,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("dVxdy",fieldsolver::dVxdy,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("dVxdz",fieldsolver::dVxdz,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("dVydx",fieldsolver::dVydx,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("dVydy",fieldsolver::dVydy,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("dVydz",fieldsolver::dVydz,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("dVzdx",fieldsolver::dVzdx,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("dVzdy",fieldsolver::dVzdy,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorDerivatives("dVzdz",fieldsolver::dVzdz,1));
      }
      if(*it == "BVOLderivs") {
         outputReducer->addOperator(new DRO::DataReductionOperatorBVOLDerivatives("dPERBXVOLdy",bvolderivatives::dPERBXVOLdy,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorBVOLDerivatives("dBGBXVOLdy",bvolderivatives::dBGBXVOLdy,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorBVOLDerivatives("dPERBXVOLdz",bvolderivatives::dPERBXVOLdz,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorBVOLDerivatives("dBGBXVOLdz",bvolderivatives::dBGBXVOLdz,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorBVOLDerivatives("dPERBYVOLdx",bvolderivatives::dPERBYVOLdx,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorBVOLDerivatives("dBGBYVOLdx",bvolderivatives::dBGBYVOLdx,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorBVOLDerivatives("dPERBYVOLdz",bvolderivatives::dPERBYVOLdz,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorBVOLDerivatives("dBGBYVOLdz",bvolderivatives::dBGBYVOLdz,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorBVOLDerivatives("dPERBZVOLdx",bvolderivatives::dPERBZVOLdx,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorBVOLDerivatives("dBGBZVOLdx",bvolderivatives::dBGBZVOLdx,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorBVOLDerivatives("dPERBZVOLdy",bvolderivatives::dPERBZVOLdy,1));
         outputReducer->addOperator(new DRO::DataReductionOperatorBVOLDerivatives("dBGBZVOLdy",bvolderivatives::dBGBZVOLdy,1));
      }
      if (*it == "Potential") {
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("poisson/potential",CellParams::PHI,1));
      }
      if (*it == "ChargeDensity") {
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("poisson/rho_q",CellParams::RHOQ_TOT,1));
      }
      if (*it == "PotentialError") {
         outputReducer->addOperator(new DRO::DataReductionOperatorCellParams("poisson/pot_error",CellParams::PHI_TMP,1));
      }
   }

   for (it = P::diagnosticVariableList.begin();
        it != P::diagnosticVariableList.end();
        it++) {
      if(*it == "FluxB")
         diagnosticReducer->addOperator(new DRO::DiagnosticFluxB);
      if(*it == "FluxE")
         diagnosticReducer->addOperator(new DRO::DiagnosticFluxE);
      if(*it == "Blocks")
         diagnosticReducer->addOperator(new DRO::Blocks);
      if(*it == "Pressure")
         diagnosticReducer->addOperator(new DRO::VariablePressure);
      if(*it == "Rho")
         diagnosticReducer->addOperator(new DRO::DataReductionOperatorCellParams("rho",CellParams::RHO,1));
      if(*it == "RhoLossAdjust")
         diagnosticReducer->addOperator(new DRO::DataReductionOperatorCellParams("rho_loss_adjust",CellParams::RHOLOSSADJUST,1));
      if(*it == "RhoLossVelBoundary")
         diagnosticReducer->addOperator(new DRO::DataReductionOperatorCellParams("rho_loss_velocity_boundary",CellParams::RHOLOSSVELBOUNDARY,1));
      if(*it == "LBweight")
         diagnosticReducer->addOperator(new DRO::DataReductionOperatorCellParams("LB_weight",CellParams::LBWEIGHTCOUNTER,1));
      if(*it == "MaxVdt")
         diagnosticReducer->addOperator(new DRO::DataReductionOperatorCellParams("max_v_dt",CellParams::MAXVDT,1));
      if(*it == "MaxRdt")
         diagnosticReducer->addOperator(new DRO::DataReductionOperatorCellParams("max_r_dt",CellParams::MAXRDT,1));
      if(*it == "MaxFieldsdt")
         diagnosticReducer->addOperator(new DRO::DataReductionOperatorCellParams("max_fields_dt",CellParams::MAXFDT,1));
      if(*it == "MaxDistributionFunction")
         diagnosticReducer->addOperator(new DRO::MaxDistributionFunction);
      if(*it == "MinDistributionFunction")
         diagnosticReducer->addOperator(new DRO::MinDistributionFunction);
      if(*it == "BoundaryType")
         diagnosticReducer->addOperator(new DRO::BoundaryType);
      if(*it == "BoundaryLayer")
         diagnosticReducer->addOperator(new DRO::BoundaryLayer);
   }
   
   for( it = P::populationMergerVariableList.begin();
        it != P::populationMergerVariableList.end();
        ++it ) {
      if( *it == "Rho" )
        populationReducer->addOperator(new DRO::VariablePopulationRho);
   }
}

// ************************************************************
// ***** DEFINITIONS FOR DATAREDUCER CLASS *****
// ************************************************************

/** Constructor for class DataReducer.
 */
DataReducer::DataReducer() { }

/** Destructor for class DataReducer. All stored DataReductionOperators 
 * are deleted.
 */
DataReducer::~DataReducer() {
   // Call delete for each DataReductionOperator:
   for (vector<DRO::DataReductionOperator*>::iterator it=operators.begin(); it!=operators.end(); ++it) {
      delete *it;
      *it = NULL;
   }
}

/** Add a new DRO::DataReductionOperator which has been created with new operation. 
 * DataReducer will take care of deleting it.
 * @return If true, the given DRO::DataReductionOperator was added successfully.
 */
bool DataReducer::addOperator(DRO::DataReductionOperator* op) {
   operators.push_back(op);
   return true;
}

/** Get the name of a DataReductionOperator.
 * @param operatorID ID number of the operator whose name is requested.
 * @return Name of the operator.
 */
std::string DataReducer::getName(const unsigned int& operatorID) const {
   if (operatorID >= operators.size()) return "";
   return operators[operatorID]->getName();
}

/** Get the name of a DataReductionOperator.
 * @param operatorID ID number of the operator whose name is requested.
 * @param population Population ID for population reducer
 * @return Name of the operator.
 */
std::string DataReducer::getName(const unsigned int& operatorID, const int population) const {
   if (operatorID >= operators.size()) return "";
   return operators[operatorID]->getName(population);
}


/** Get info on the type of data calculated by the given DataReductionOperator.
 * A DataReductionOperator writes an array on disk. Each element of the array is a vector with n elements. Finally, each
 * vector element has a byte size, as given by the sizeof function.
 * @param operatorID ID number of the DataReductionOperator whose output data info is requested.
 * @param dataType Basic datatype, must be int, uint, or float.
 * @param dataSize Byte size of written datatype, for example double-precision floating points
 * have byte size of sizeof(double).
 * @param vectorSize How many elements are in the vector returned by the DataReductionOperator.
 * @return If true, DataReductionOperator was found and it returned sensible values.
 */
bool DataReducer::getDataVectorInfo(const unsigned int& operatorID,std::string& dataType,unsigned int& dataSize,unsigned int& vectorSize) const {
   if (operatorID >= operators.size()) return false;
   return operators[operatorID]->getDataVectorInfo(dataType,dataSize,vectorSize);
}

/** Request a DataReductionOperator to calculate its output data and to write it to the given buffer.
 * @param cell Pointer to spatial cell whose data is to be reduced.
 * @param operatorID ID number of the applied DataReductionOperator.
 * @param buffer Buffer in which DataReductionOperator should write its data.
 * @return If true, DataReductionOperator calculated and wrote data successfully.
 */
bool DataReducer::reduceData(const SpatialCell* cell,const unsigned int& operatorID,char* buffer) {
   // Tell the chosen operator which spatial cell we are counting:
   if (operatorID >= operators.size()) return false;
   if (operators[operatorID]->setSpatialCell(cell) == false) return false;

   if (operators[operatorID]->reduceData(cell,buffer) == false) return false;
   return true;
}

/** Request a DataReductionOperator to calculate its output data and to write it to the given buffer.
 * @param cell Pointer to spatial cell whose data is to be reduced.
 * @param operatorID ID number of the applied DataReductionOperator.
 * @param population Population ID for population reducer
 * @param buffer Buffer in which DataReductionOperator should write its data.
 * @return If true, DataReductionOperator calculated and wrote data successfully.
 */
bool DataReducer::reduceData(const SpatialCell* cell,const unsigned int& operatorID, const int population, char* buffer) {
   // Tell the chosen operator which spatial cell we are counting:
   if (operatorID >= operators.size()) return false;
   if (operators[operatorID]->setSpatialCell(cell) == false) return false;

   if (operators[operatorID]->reduceData(cell,population,buffer) == false) return false;
   return true;
}

/** Request a DataReductionOperator to calculate its output data and to write it to the given variable.
 * @param cell Pointer to spatial cell whose data is to be reduced.
 * @param operatorID ID number of the applied DataReductionOperator.
 * @param result Real variable in which DataReductionOperator should write its result.
 * @return If true, DataReductionOperator calculated and wrote data successfully.
 */
bool DataReducer::reduceData(const SpatialCell* cell,const unsigned int& operatorID,Real * result) {
   // Tell the chosen operator which spatial cell we are counting:
   if (operatorID >= operators.size()) return false;
   if (operators[operatorID]->setSpatialCell(cell) == false) return false;
   
   if (operators[operatorID]->reduceData(cell,result) == false) return false;
   return true;
}

/** Get the number of DataReductionOperators stored in DataReducer.
 * @return Number of DataReductionOperators stored in DataReducer.
 */
unsigned int DataReducer::size() const {return operators.size();}

