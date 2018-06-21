#pragma once
#include <string>
#include <map>
#include <vector>
using namespace std;
class TestBase
{
public:
	virtual void operator() (string name, map<string, double> num_vars, map<string,string> str_vars, map<string, vector<double>> list_vars) = 0;
};

