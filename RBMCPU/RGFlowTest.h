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
#include "TestBase.h"

class RGFlowTest : public TestBase
{

public:
	RGFlowTest();
	~RGFlowTest();
	void plotConvergence(double beta);
	void plotError(vector<int> num_samples);
	void plotRGFlow(double startingBeta);
	void plotRGFlowNew(double startingBeta, int batch_size);
	void plotRGFlowLamNeq0(double startingBeta, double startingLam, int batch_size, int chain_size=512, int layer_size=8, int maxiterations=400);

	void modTest(double startingBeta);
	void testGibbsConvergence();
	void cheatTest(double startingBeta);
	

	// Inherited via TestBase
	virtual void operator()(string name, map<string, double> num_vars, map<string, string> str_vars, map<string, vector<double>> list_vars) override;

};

