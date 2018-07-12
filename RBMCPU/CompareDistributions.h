#pragma once
#include "TestBase.h"
#include "Phi1D.h"
#include "RBMCompTree.h"
#include "ContrastiveDivergence.h"
#include "Session.h"
class CompareDistributions :
	public TestBase
{
public:
	CompareDistributions();
	~CompareDistributions();
	void runTest(double kappa, double lambda, double lr, int chainsize, int batchsize);
	// Inherited via TestBase
	virtual void operator()(string name, map<string, double> num_vars, map<string, string> str_vars, map<string, vector<double>> list_vars) override;
};

