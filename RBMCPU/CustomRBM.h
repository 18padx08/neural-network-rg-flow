#pragma once
#include <vector>
#include <string>

using namespace std;
class CustomRBM
{
public:
	CustomRBM();
	~CustomRBM();
	void setWeights(vector<double> weights);
	void parseConfig(string filename);
};

