#pragma once
#include <omp.h>
#include "../CudaTest/Ising1D.h"
#include "MNISTTest.h"
#include "SymmetryTest.h"
#include "TIRBMTest.h"
#include "AnalyticalTest.h"
#include "RG.h"
#include "RBM.h"
#include "targetver.h"
#include <stdio.h>
//#include <tchar.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include "Session.h"
#include "Graph.h"
#include "Add.h"
#include "Sigmoid.h"
#include "RGLayer.h"
#include "ProbPooling.h"
#include "RBMCompTree.h"
#include "ContrastiveDivergence.h"
#include "ModCD.h"
#include "CheatCD.h"

using namespace ct;

class ErrorAnalysis
{
private:
	double errorMonteCarlo(shared_ptr<Tensor> samples);
public:
	ErrorAnalysis();
	~ErrorAnalysis();

	//use one batch to train the network and compare the monte carlo error to the network output error
	//important quantities include visible correlation and hidden correlation and network parameter
	//also check reconstruction error
	void plotErrorOnTraining(double beta = 1.0, int bs=10);

	//use a pretrained network (e.g. calculate the parameter analytically) and check the response error
	void plotErrorOfResponse(double beta = 1.0);

	//lambda != 0 tests
	void plotNonZeroLamTests(double kappa = 0.4, double lambda = 1);

};

