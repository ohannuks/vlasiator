/*
This file is part of Vlasiator.

Copyright 2011, 2012 Finnish Meteorological Institute












*/

#ifndef LARMOR_H
#define LARMOR_H

#include "definitions.h"
#include "spatial_cell.hpp"
#include "projects/projects_common.h"
#include "projects/projects_vlasov_acceleration.h"

#include "dccrg.hpp"

struct larmorParameters {
   static Real BX0;
   static Real BY0;
   static Real BZ0;
   static Real VX0;
   static Real VY0;
   static Real VZ0;
   static Real X0;
   static Real Y0;
   static Real Z0;
   static Real DENSITY;

} ;

/**
 * Initialize project.
 */
bool initializeProject(void);

/** Register parameters that should be read in
 */
bool addProjectParameters(void);
/** Get the value that was read in
 */
bool getProjectParameters(void);

/*!\brief Set the fields and distribution of a cell according to the default simulation settings.
 * This is used for the NOT_SYSBOUNDARY cells and some other system boundary conditions (e.g. Outflow).
 * \param cell Pointer to the cell to set.
 */
void setProjectCell(SpatialCell* cell);

template<typename UINT,typename REAL> void calcAccFaceX(
   REAL& ax, REAL& ay, REAL& az,
   const UINT& I, const UINT& J, const UINT& K,
   const REAL* const cellParams,
   const REAL* const blockParams,
   const REAL* const cellBVOLDerivatives
) {
   lorentzForceFaceX(ax,ay,az,I,J,K,cellParams,blockParams,cellBVOLDerivatives);
}

template<typename UINT,typename REAL> void calcAccFaceY(
   REAL& ax, REAL& ay, REAL& az,
   const UINT& I, const UINT& J, const UINT& K,
   const REAL* const cellParams,
   const REAL* const blockParams,
   const REAL* const cellBVOLDerivatives
) {
   lorentzForceFaceY(ax,ay,az,I,J,K,cellParams,blockParams,cellBVOLDerivatives);
}

template<typename UINT,typename REAL> void calcAccFaceZ(
   REAL& ax, REAL& ay, REAL& az,
   const UINT& I, const UINT& J, const UINT& K,
   const REAL* const cellParams,
   const REAL* const blockParams,
   const REAL* const cellBVOLDerivatives
) {
   lorentzForceFaceZ(ax,ay,az,I,J,K,cellParams,blockParams,cellBVOLDerivatives);
}

#endif
