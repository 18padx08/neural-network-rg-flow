#include "Phi2DMCTests.h"



Phi2DMCTests::Phi2DMCTests()
{
}


Phi2DMCTests::~Phi2DMCTests()
{
}

void Phi2DMCTests::criticalLineTest(vector<int> chainsize)
{
	ofstream output("phi2dmctest.csv");
	for (double l : {0.0, 0.1, 0.5, 1.0}) {
		for (double k : {0.3,0.4,0.5,0.8,1.1,1.4}) {
			Phi2D phi(chainsize, k, l);
			phi.thermalize();
			double absAvg = 0;
			for (int i = 0; i < 50; i++) {
				absAvg += abs(phi.volumeAverage());
				phi.monteCarloSweep();
			}
			absAvg /= 50;
			output << l << "," << k << "," << absAvg << std::endl;
			if (absAvg > 0.1) break;
			std::cout << "At l=" << l << " k=" << k << " with vev=" << absAvg << std::endl;
		}
	}
}

void Phi2DMCTests::operator()(string name, map<string, double> num_vars, map<string, string> str_vars, map<string, vector<double>> list_vars)
{
	vector<int> chainsize = this->getIntVector("chainsize", num_vars, list_vars);
	if (name == "criticalLineTest") {
		criticalLineTest(chainsize);
	}
}
