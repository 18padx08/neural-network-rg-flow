#pragma once
#include "Phi1D.h"
#include <fstream>
#include "TestBase.h"
#include "RBMCompTree.h"
#include "ContrastiveDivergence.h"
#include "Session.h"
#include "Storage.h"
class TestConvergence : public TestBase
{
public:
	TestConvergence();
	~TestConvergence();
	void testConvergence(double learningRate,double kappa, double lambda, int chainsize, int batchsize, bool useZ2);
	void extractFromHidden(double learningRate, double kappa, double lambda, int chainsize, int batchsize, bool useZ2);
	// Inherited via TestBase
	virtual void operator()(string name, map<string, double> num_vars, map<string, string> str_vars, map<string, vector<double>> list_vars) override;
};

