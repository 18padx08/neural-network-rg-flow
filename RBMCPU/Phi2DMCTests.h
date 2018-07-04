#pragma once
#include "TestBase.h"
#include "Phi2D.h"
#include <fstream>
class Phi2DMCTests : public TestBase
{
public:
	Phi2DMCTests();
	~Phi2DMCTests();

	void criticalLineTest(vector<int> chainsize, vector<double> kappas, vector<double> lambdas);
	// Inherited via TestBase
	virtual void operator()(string name, map<string, double> num_vars, map<string, string> str_vars, map<string, vector<double>> list_vars) override;

};

