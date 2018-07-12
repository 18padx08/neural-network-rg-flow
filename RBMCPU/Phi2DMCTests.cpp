#include "Phi2DMCTests.h"



Phi2DMCTests::Phi2DMCTests()
{
}


Phi2DMCTests::~Phi2DMCTests()
{
}

void Phi2DMCTests::criticalLineTest(vector<int> chainsize, vector<double> kappas, vector<double> lambdas)
{
#pragma omp parallel for
	for (int i = 0; i < lambdas.size(); i++) {
		double l = lambdas[i];
		double k = kappas[0];
		int counter = 0;
		ofstream output("phi2dmctest_lambda=" + to_string(l) + ".csv");
		//take first value as base
		double baseValue = 0.1;
		bool first = true;
		while(true) {
			Phi2D phi(chainsize, k, l);
			phi.useWolff = true;
			phi.thermalize();
			
			double absAvg = 0;
			for (int i = 0; i < 100; i++) {
				absAvg += abs(phi.volumeAverage());
				phi.monteCarloSweep();
			}
			absAvg /= 100;
			if (first)
			{
				baseValue = absAvg*10;
				first = false;

			}
			output << l << "," << k << "," << absAvg << std::endl;
			std::cout << "At l=" << l << " k=" << k << " with vev=" << absAvg << std::endl;
			
			if (absAvg > baseValue) {
				counter++;
				if (counter > 5) break;
			}
			k += 0.02;
			
		}
	}
}

void Phi2DMCTests::criticalLineTestNN(vector<int> chainsize, vector<double> kappas, vector<double> lambdas)
{
#pragma omp parallel for
	for (int i = 0; i < lambdas.size(); i++) {
		double l = lambdas[i];
		double k = kappas[0];
		int counter = 0;
		ofstream output("phi2dmctestNN_lambda=" + to_string(l) + ".csv");
		//take first value as base
		double baseValue = 0.1;
		bool first = true;
		while (true) {
			auto graph = RBMCompTree::getRBM2DGraph();
			auto kappa = graph->getVarForName("kappa");
			auto lambda = graph->getVarForName("lambda");
			kappa->value = make_shared<Tensor>(Tensor({ 1 }, { k }));
			lambda->value = make_shared<Tensor>(Tensor({ 1 }, { l }));

			shared_ptr<Session> session = make_shared<Session>(graph);
			map<string, shared_ptr<Tensor>> feedDic;
			vector<double> samples(100 * chainsize[0]*chainsize[1]);
			vector<int> dims = { chainsize[0],chainsize[1],100 };
			feedDic = { {"x", make_shared<Tensor>(dims,samples)} };
			session->run(feedDic, true, 50);
			double absAvg = 0;
			auto visibles = *(*dynamic_pointer_cast<ct::Storage>(graph->storages["visibles_raw"].lock())).storage[50];
			auto hiddens = *(*dynamic_pointer_cast<ct::Storage>(graph->storages["hiddens_raw"].lock())).storage[50];
#pragma omp parallel for reduction(+:absAvg)
			for (int b = 0; b < 100; b++) {
				double tmpAvg = 0;
#pragma omp parallel for reduction(+:tmpAvg)
				for (int i = 0; i < hiddens.dimensions[0]; i++) {
#pragma omp parallel for reduction(+:tmpAvg)
					for (int j = 0; j < hiddens.dimensions[1]; j++)
					{
						tmpAvg += (visibles[{2 * i, 2 * j, b}] + hiddens[{i, j, b}]);
						
					}
				}
				tmpAvg /= visibles.dimensions[0] * visibles.dimensions[1];
				absAvg += abs(tmpAvg);
			}
			absAvg /= 100;
			if (first)
			{
				baseValue = absAvg * 10;
				first = false;

			}
			output << l << "," << k << "," << absAvg << std::endl;
			std::cout << "At l=" << l << " k=" << k << " with vev=" << absAvg << std::endl;

			if (absAvg > baseValue) {
				counter++;
				if (counter > 5) break;
			}
			k += 0.02;

		}
	}
}

void Phi2DMCTests::operator()(string name, map<string, double> num_vars, map<string, string> str_vars, map<string, vector<double>> list_vars)
{
	vector<int> chainsize = this->getIntVector("chainsize", num_vars, list_vars);
	vector<double> kappas = this->getDoubleVector("kappa", num_vars, list_vars);
	vector<double> lambdas = this->getDoubleVector("lambda", num_vars, list_vars);
	if (name == "criticalLineTest") {
		criticalLineTest(chainsize, kappas, lambdas);
	}
	if (name == "criticalLineNNTest") {
		criticalLineTestNN(chainsize, kappas, lambdas);
	}
}
