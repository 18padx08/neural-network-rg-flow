#pragma once
#include <string>
#include <map>
#include <vector>
using namespace std;
class TestBase
{
public:
	virtual void operator() (string name, map<string, double> num_vars, map<string,string> str_vars, map<string, vector<double>> list_vars) = 0;

	vector<double> getDoubleVector(string name, map<string, double> num_vars, map<string, vector<double>> list_vars) {
		vector<double> tmp;
		if (list_vars.find(name) == list_vars.end()) {
			if (num_vars.find(name) == num_vars.end()) {
				return vector<double>();
			}
			tmp.push_back(num_vars.at(name));
		}
		else {
			tmp = list_vars.at(name);
		}
		return tmp;
	}

	vector<int> getIntVector(string name, map<string, double> num_vars, map<string, vector<double>> list_vars) {
		vector<int> tmp;
		if (list_vars.find(name) == list_vars.end()) {
			if (num_vars.find(name) == num_vars.end()) {
				return vector<int>();
			}
			tmp.push_back(num_vars.at(name));
		}
		else {
			auto tmp2 = list_vars.at(name);
			for (auto t : tmp2) {
				tmp.push_back(t);
			}
		}
		return tmp;
	}
};

