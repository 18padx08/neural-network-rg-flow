#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include "ErrorAnalysis.h"
#include "RGFlowTest.h"
#include <regex>
using namespace std;

 enum REGISTERED_TESTS {
	 None,
	 plotRGFlowLamNeq0,
	 plotNonZeroLamTests
};


class Config
{
private:
	int level = -1;
	REGISTERED_TESTS enumFromString(string str);
	function<void()> getFunction(REGISTERED_TESTS currentTest, map<string, double> num_vars, map<string, string> str_vars, map<string, vector<double>> list_vars);

	//patterns
	regex sectionNames;
	regex number;
	regex list;
	regex varName;
	string varNameReg;
	string numberReg;
	string listReg;
	regex end;

	vector<function<void()>> functions;
public:
	ifstream config_file;
	Config();
	Config(string config_file);
	~Config();

	void load();
	void run();

};

