#pragma once
#include "TestBase.h"
#include "Phi2D.h"
#include "Tensor.h"
#include "RBMCompTree.h"
#include "ContrastiveDivergence2D.h"
#include "Session.h"

using namespace ct;
using namespace ct::optimizers;
class RG2DTest :
	public TestBase
{
public:
	RG2DTest();
	~RG2DTest();

	void run(vector<int> size, int batchsize, double kappa, double lambda, double lr);
	void test2dConvergence(vector<int> size, int batchsize, double kappa, double lambda, double lr);
	// Inherited via TestBase
	virtual void operator()(string name, map<string, double> num_vars, map<string, string> str_vars, map<string, vector<double>> list_vars) override;
};

