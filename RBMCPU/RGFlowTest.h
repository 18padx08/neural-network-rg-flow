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
#include <tchar.h>
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

class RGFlowTest
{
public:
	RGFlowTest();
	~RGFlowTest();
	void plotConvergence(double beta);
	void plotError(vector<int> num_samples);
	void plotRGFlow(double startingBeta);
	void plotRGFlowNew(double startingBeta);

	void modTest(double startingBeta);
	void testGibbsConvergence();
	void cheatTest(double startingBeta);
};

