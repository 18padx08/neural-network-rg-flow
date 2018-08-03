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
#include "Phi1D.h"
#include "TestBase.h"
class NormalizationTests : public TestBase
{
public:
	NormalizationTests();
	~NormalizationTests();
	void run();
	void runConvTest();
	void compareLatticeAndNN();
	void compareNormOverVariousKappa(vector<double> kappas, int chainsize);

	// Inherited via TestBase
	virtual void operator()(string name, map<string, double> num_vars, map<string, string> str_vars, map<string, vector<double>> list_vars) override;
};

