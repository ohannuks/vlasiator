/*
 * This file is part of Vlasiator.
 * 
 * Copyright 2012 Finnish Meteorological Institute
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

#include "../common.h"
#include "../definitions.h"
#include "../parameters.h"
#include "cmath"
#include "backgroundfield.h"
#include "fieldfunction.hpp"
#include "integratefunction.hpp"

//FieldFunction should be initialized
void setBackgroundField(
   FieldFunction& bgFunction,
   Real* cellParams,
   Real* faceDerivatives,
   Real* volumeDerivatives,
   bool append) {
   using namespace CellParams;
   using namespace fieldsolver;
   using namespace bvolderivatives;
   
   //these are doubles, as the averaging functions copied from Gumics
   //use internally doubles. In any case, it should provide more
   //accurate results also for float simulations
   double accuracy = 1e-17;
   double start[3];
   double end[3];
   double dx[3];
   unsigned int faceCoord1[3];
   unsigned int faceCoord2[3];


   start[0] = cellParams[CellParams::XCRD];
   start[1] = cellParams[CellParams::YCRD];
   start[2] = cellParams[CellParams::ZCRD];

   dx[0] = cellParams[CellParams::DX];
   dx[1] = cellParams[CellParams::DY];
   dx[2] = cellParams[CellParams::DZ];

   end[0]=start[0]+dx[0];
   end[1]=start[1]+dx[1];
   end[2]=start[2]+dx[2];
   
   //the coordinates of the edges face with a normal in the third coordinate direction, stored here to enable looping
   faceCoord1[0]=1;
   faceCoord2[0]=2;
   faceCoord1[1]=0;
   faceCoord2[1]=2;
   faceCoord1[2]=0;
   faceCoord2[2]=1;

   /*if we do not add a new background to the existing one we first put everything to zero*/
   if(append==false) {
      setBackgroundFieldToZero(cellParams, faceDerivatives, volumeDerivatives);
   }
   
   //Face averages
   for(unsigned int fComponent=0;fComponent<3;fComponent++){
      bgFunction.setDerivative(0);
      bgFunction.setComponent((coordinate)fComponent);
      cellParams[CellParams::BGBX+fComponent] += 
      surfaceAverage(
         bgFunction,
         (coordinate)fComponent,
         accuracy,
         start,
         dx[faceCoord1[fComponent]],
         dx[faceCoord2[fComponent]]
      );
      
      //Compute derivatives. Note that we scale by dx[] as the arrays are assumed to contain differences, not true derivatives!
      bgFunction.setDerivative(1);
      bgFunction.setDerivComponent((coordinate)faceCoord1[fComponent]);
      faceDerivatives[fieldsolver::dBGBxdy+2*fComponent] +=
         dx[faceCoord1[fComponent]]*
         surfaceAverage(bgFunction,(coordinate)fComponent,accuracy,start,dx[faceCoord1[fComponent]],dx[faceCoord2[fComponent]]);
      bgFunction.setDerivComponent((coordinate)faceCoord2[fComponent]);
      faceDerivatives[fieldsolver::dBGBxdy+1+2*fComponent] +=
         dx[faceCoord2[fComponent]]*
         surfaceAverage(bgFunction,(coordinate)fComponent,accuracy,start,dx[faceCoord1[fComponent]],dx[faceCoord2[fComponent]]);
   }

   //Volume averages
   for(unsigned int fComponent=0;fComponent<3;fComponent++){
      bgFunction.setDerivative(0);
      bgFunction.setComponent((coordinate)fComponent);
      cellParams[CellParams::BGBXVOL+fComponent] += volumeAverage(bgFunction,accuracy,start,end);

      //Compute derivatives. Note that we scale by dx[] as the arrays are assumed to contain differences, not true derivatives!      
      bgFunction.setDerivative(1);
      bgFunction.setDerivComponent((coordinate)faceCoord1[fComponent]);
      volumeDerivatives[bvolderivatives::dBGBXVOLdy+2*fComponent] +=  dx[faceCoord1[fComponent]]*volumeAverage(bgFunction,accuracy,start,end);
      bgFunction.setDerivComponent((coordinate)faceCoord2[fComponent]);
      volumeDerivatives[bvolderivatives::dBGBXVOLdy+1+2*fComponent] += dx[faceCoord2[fComponent]]*volumeAverage(bgFunction,accuracy,start,end);
   }
   
   // Edge averages
   // As of 20131115, these components are only needed in the Hall term calculations.
   bgFunction.setDerivative(0);
   if(Parameters::ohmHallTerm > 0) {
      start[0] = cellParams[CellParams::XCRD];
      start[1] = cellParams[CellParams::YCRD];
      start[2] = cellParams[CellParams::ZCRD];
      bgFunction.setComponent(X);
      cellParams[CellParams::BGBX_000_010] +=
         lineAverage(
            bgFunction,
            Y,
            accuracy,
            start,
            dx[1]
         );
      cellParams[CellParams::BGBX_000_001] +=
         lineAverage(
            bgFunction,
            Z,
            accuracy,
            start,
            dx[2]
         );
      
      bgFunction.setComponent(Y);
      cellParams[CellParams::BGBY_000_100] +=
         lineAverage(
            bgFunction,
            X,
            accuracy,
            start,
            dx[0]
         );
      cellParams[CellParams::BGBY_000_001] +=
         lineAverage(
            bgFunction,
            Z,
            accuracy,
            start,
            dx[2]
         );
      
      bgFunction.setComponent(Z);
      cellParams[CellParams::BGBZ_000_100] +=
         lineAverage(
            bgFunction,
            X,
            accuracy,
            start,
            dx[0]
         );
      cellParams[CellParams::BGBZ_000_010] +=
         lineAverage(
            bgFunction,
            Y,
            accuracy,
            start,
            dx[1]
         );
      
      start[0] = cellParams[CellParams::XCRD] + cellParams[CellParams::DX];
      start[1] = cellParams[CellParams::YCRD];
      start[2] = cellParams[CellParams::ZCRD];
      bgFunction.setComponent(X);
      cellParams[CellParams::BGBX_100_110] +=
         lineAverage(
            bgFunction,
            Y,
            accuracy,
            start,
            dx[1]
         );
      cellParams[CellParams::BGBX_100_101] +=
         lineAverage(
            bgFunction,
            Z,
            accuracy,
            start,
            dx[2]
         );
      
      bgFunction.setComponent(Y);
      cellParams[CellParams::BGBY_100_101] +=
         lineAverage(
            bgFunction,
            Z,
            accuracy,
            start,
            dx[2]
         );
      
      bgFunction.setComponent(Z);
      cellParams[CellParams::BGBZ_100_110] +=
         lineAverage(
            bgFunction,
            Y,
            accuracy,
            start,
            dx[1]
         );
      
      start[0] = cellParams[CellParams::XCRD];
      start[1] = cellParams[CellParams::YCRD];
      start[2] = cellParams[CellParams::ZCRD] + cellParams[CellParams::DZ];
      bgFunction.setComponent(X);
      cellParams[CellParams::BGBX_001_011] +=
         lineAverage(
            bgFunction,
            Y,
            accuracy,
            start,
            dx[1]
         );
      
      bgFunction.setComponent(Y);
      cellParams[CellParams::BGBY_001_101] +=
         lineAverage(
            bgFunction,
            X,
            accuracy,
            start,
            dx[0]
         );
      
      bgFunction.setComponent(Z);
      cellParams[CellParams::BGBZ_001_011] +=
         lineAverage(
            bgFunction,
            Y,
            accuracy,
            start,
            dx[1]
         );
      
      start[0] = cellParams[CellParams::XCRD] + cellParams[CellParams::DX];
      start[1] = cellParams[CellParams::YCRD];
      start[2] = cellParams[CellParams::ZCRD] + cellParams[CellParams::DZ];
      bgFunction.setComponent(X);
      cellParams[CellParams::BGBX_101_111] +=
         lineAverage(
            bgFunction,
            Y,
            accuracy,
            start,
            dx[1]
         );
      
      bgFunction.setComponent(Z);
      cellParams[CellParams::BGBZ_101_111] +=
         lineAverage(
            bgFunction,
            Y,
            accuracy,
            start,
            dx[1]
         );
      
      start[0] = cellParams[CellParams::XCRD];
      start[1] = cellParams[CellParams::YCRD] + cellParams[CellParams::DY];
      start[2] = cellParams[CellParams::ZCRD];
      bgFunction.setComponent(X);
      cellParams[CellParams::BGBX_010_011] +=
         lineAverage(
            bgFunction,
            Z,
            accuracy,
            start,
            dx[2]
         );
      
      bgFunction.setComponent(Y);
      cellParams[CellParams::BGBY_010_110] +=
         lineAverage(
            bgFunction,
            X,
            accuracy,
            start,
            dx[0]
         );
      cellParams[CellParams::BGBY_010_011] +=
         lineAverage(
            bgFunction,
            Z,
            accuracy,
            start,
            dx[2]
         );
      
      bgFunction.setComponent(Z);
      cellParams[CellParams::BGBZ_010_110] +=
         lineAverage(
            bgFunction,
            X,
            accuracy,
            start,
            dx[0]
         );
      
      start[0] = cellParams[CellParams::XCRD] + cellParams[CellParams::DX];
      start[1] = cellParams[CellParams::YCRD] + cellParams[CellParams::DY];
      start[2] = cellParams[CellParams::ZCRD];
      bgFunction.setComponent(X);
      cellParams[CellParams::BGBX_110_111] +=
         lineAverage(
            bgFunction,
            Z,
            accuracy,
            start,
            dx[2]
         );
      
      bgFunction.setComponent(Y);
      cellParams[CellParams::BGBY_110_111] +=
         lineAverage(
            bgFunction,
            Z,
            accuracy,
            start,
            dx[2]
         );
      
      start[0] = cellParams[CellParams::XCRD];
      start[1] = cellParams[CellParams::YCRD] + cellParams[CellParams::DY];
      start[2] = cellParams[CellParams::ZCRD] + cellParams[CellParams::DZ];
      bgFunction.setComponent(Y);
      cellParams[CellParams::BGBY_011_111] +=
         lineAverage(
            bgFunction,
            X,
            accuracy,
            start,
            dx[0]
         );
      
      bgFunction.setComponent(Z);
      cellParams[CellParams::BGBZ_011_111] +=
         lineAverage(
            bgFunction,
            X,
            accuracy,
            start,
            dx[0]
         );
      
      start[0] = cellParams[CellParams::XCRD];
      start[1] = cellParams[CellParams::YCRD];
      start[2] = cellParams[CellParams::ZCRD] + cellParams[CellParams::DZ];
      bgFunction.setComponent(Z);
      cellParams[CellParams::BGBZ_001_101] +=
         lineAverage(
            bgFunction,
            X,
            accuracy,
            start,
            dx[0]
         );
   } else {
      cellParams[CellParams::BGBX_000_010] += 0.0;
      cellParams[CellParams::BGBX_100_110] += 0.0;
      cellParams[CellParams::BGBX_001_011] += 0.0;
      cellParams[CellParams::BGBX_101_111] += 0.0;
      cellParams[CellParams::BGBX_000_001] += 0.0;
      cellParams[CellParams::BGBX_100_101] += 0.0;
      cellParams[CellParams::BGBX_010_011] += 0.0;
      cellParams[CellParams::BGBX_110_111] += 0.0;
      cellParams[CellParams::BGBY_000_100] += 0.0;
      cellParams[CellParams::BGBY_010_110] += 0.0;
      cellParams[CellParams::BGBY_001_101] += 0.0;
      cellParams[CellParams::BGBY_011_111] += 0.0;
      cellParams[CellParams::BGBY_000_001] += 0.0;
      cellParams[CellParams::BGBY_100_101] += 0.0;
      cellParams[CellParams::BGBY_010_011] += 0.0;
      cellParams[CellParams::BGBY_110_111] += 0.0;
      cellParams[CellParams::BGBZ_000_100] += 0.0;
      cellParams[CellParams::BGBZ_010_110] += 0.0;
      cellParams[CellParams::BGBZ_001_101] += 0.0;
      cellParams[CellParams::BGBZ_011_111] += 0.0;
      cellParams[CellParams::BGBZ_000_010] += 0.0;
      cellParams[CellParams::BGBZ_100_110] += 0.0;
      cellParams[CellParams::BGBZ_001_011] += 0.0;
      cellParams[CellParams::BGBZ_101_111] += 0.0;
   }

   //TODO
   //COmpute divergence and curl of volume averaged field and check that both are zero. 
}

void setBackgroundFieldToZero(
   Real* cellParams,
   Real* faceDerivatives,
   Real* volumeDerivatives
) {
   using namespace CellParams;
   using namespace fieldsolver;
   using namespace bvolderivatives;
   
   //Face averages
   for(unsigned int fComponent=0;fComponent<3;fComponent++){
      cellParams[CellParams::BGBX+fComponent] = 0.0;
      faceDerivatives[fieldsolver::dBGBxdy+2*fComponent] = 0.0;
      faceDerivatives[fieldsolver::dBGBxdy+1+2*fComponent] = 0.0;
   }
   
   //Volume averages
   for(unsigned int fComponent=0;fComponent<3;fComponent++){
      cellParams[CellParams::BGBXVOL+fComponent] = 0.0;
      volumeDerivatives[bvolderivatives::dBGBXVOLdy+2*fComponent] = 0.0;
      volumeDerivatives[bvolderivatives::dBGBXVOLdy+1+2*fComponent] =0.0;
   }
   
   //Terms needed for hall term
   cellParams[CellParams::BGBX_000_010] = 0.0;
   cellParams[CellParams::BGBX_100_110] = 0.0;
   cellParams[CellParams::BGBX_001_011] = 0.0;
   cellParams[CellParams::BGBX_101_111] = 0.0;
   cellParams[CellParams::BGBX_000_001] = 0.0;
   cellParams[CellParams::BGBX_100_101] = 0.0;
   cellParams[CellParams::BGBX_010_011] = 0.0;
   cellParams[CellParams::BGBX_110_111] = 0.0;
   cellParams[CellParams::BGBY_000_100] = 0.0;
   cellParams[CellParams::BGBY_010_110] = 0.0;
   cellParams[CellParams::BGBY_001_101] = 0.0;
   cellParams[CellParams::BGBY_011_111] = 0.0;
   cellParams[CellParams::BGBY_000_001] = 0.0;
   cellParams[CellParams::BGBY_100_101] = 0.0;
   cellParams[CellParams::BGBY_010_011] = 0.0;
   cellParams[CellParams::BGBY_110_111] = 0.0;
   cellParams[CellParams::BGBZ_000_100] = 0.0;
   cellParams[CellParams::BGBZ_010_110] = 0.0;
   cellParams[CellParams::BGBZ_001_101] = 0.0;
   cellParams[CellParams::BGBZ_011_111] = 0.0;
   cellParams[CellParams::BGBZ_000_010] = 0.0;
   cellParams[CellParams::BGBZ_100_110] = 0.0;
   cellParams[CellParams::BGBZ_001_011] = 0.0;
   cellParams[CellParams::BGBZ_101_111] = 0.0;

}
