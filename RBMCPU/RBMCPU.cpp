// RBMCPU.cpp : Defines the entry point for the console application.


//TODO: 
// two spin setup -> check if correct results
// enforce Z2 symmetry
// check on lattice -> compare filters to paper


#include "MNISTTest.h"
#include "SymmetryTest.h"
#include "TIRBMTest.h"
#include "AnalyticalTest.h"
#include "RG.h"
#include "RBM.h"
#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <ctime>

int main()
{
	srand(time(NULL));
	TIRBMTest tTest;
	tTest.runTest();
	MNISTTest test;
	//test.executeRBMCPU();
    return 0;
}

