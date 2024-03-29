/*
This file is part of Vlasiator.

Copyright 2010, 2011, 2012, 2013 Finnish Meteorological Institute
*/

#ifndef DEFINITIONS_H
#define DEFINITIONS_H

# include <stdint.h>

//set floating point precision for storing the distribution function here. Default is single precision, use -DDPF to set double precision
#ifdef DPF
typedef double Realf;
#else
typedef float Realf;
#endif

//set general floating point precision here. Default is single precision, use -DDP to set double precision
#ifdef DP
typedef double Real;
typedef const double creal;
#else
typedef float Real;
typedef const float creal;
#endif

typedef const int cint;
typedef unsigned char uchar;
typedef const unsigned char cuchar;

typedef uint32_t uint;
typedef const uint32_t cuint;

typedef cuint csize;

typedef uint64_t CellID;

template<typename T> T convert(const T& number) {return number;}

/** Definition of a function that takes in a velocity block with neighbor data, 
 * and returns a number that is used to decide whether or not the block should 
 * be refined or coarsened.*/
typedef Realf (*AmrVelRefinement)(const Realf* velBlock);

#endif
