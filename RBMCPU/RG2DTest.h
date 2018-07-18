#pragma once
#include "TestBase.h"
class RG2DTest :
	public TestBase
{
public:
	RG2DTest();
	~RG2DTest();

	void run(vector<int> size, double kappa, double lambda, double lr);

	// Inherited via TestBase
	virtual void operator()(string name, map<string, double> num_vars, map<string, string> str_vars, map<string, vector<double>> list_vars) override;
};

