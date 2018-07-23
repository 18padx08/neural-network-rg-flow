#include "RG2DTest.h"



RG2DTest::RG2DTest()
{
}


RG2DTest::~RG2DTest()
{

}

void RG2DTest::run(vector<int> size, int batchsize,  double kappa, double lambda, double lr)
{
	Phi2D phi4(size,kappa,lambda);
	phi4.useWolff = true;
	phi4.thermalize();

	vector<shared_ptr<Tensor>> samples(size[0] * size[1] * batchsize);
	//create sample to start with


}

void RG2DTest::test2dConvergence(vector<int> size, int batchsize, double kappa, double lambda, double lr)
{
	Phi2D phi4(size, kappa, lambda);
	phi4.useWolff = true;
	phi4.thermalize();

	vector<double> samples(size[0] * size[1] * batchsize);
	vector<int> dims = { size[0],size[1],batchsize };
	//create sample to start with
	for (int bs = 0; bs < batchsize; bs++) {
		phi4.monteCarloSweep();
		auto config = phi4.getConfiguration();
#pragma omp parallel for
		for (int i = 0; i < size[0]; i++) {
#pragma omp parallel for
			for (int j = 0; j < size[1]; j++) {
				samples[i + j * size[0] + bs * size[0] * size[1]] = config[i][j];
			}
		}
	}

	auto graph = RBMCompTree::getRBM2DGraph();
	auto session = make_shared<Session>(graph);
	auto cd = make_shared<ContrastiveDivergence2D>(graph, lr);
	map<string, shared_ptr<Tensor>> feedDic;
	feedDic = { {"x" , make_shared<Tensor>(dims, samples)} };

	auto kap = graph->getVarForName("kappa");
	auto lamb = graph->getVarForName("lambda");
	kap->value = make_shared<Tensor>(Tensor({ 1 }, {2*kappa}));
	lamb->value = make_shared<Tensor>(Tensor({ 1 }, {0.5* lambda }));

	double avgK = 0;
	double avgL = 0;
	double lastAvgK = 0;
	double lastAvgL = 0;

	double thresholdK = 0.001;
	double thresholdL = 0.001;
	int runningCounter = 1;
	int overall = 0;
	ofstream fileToSave("2d_convergence_kappa=" + to_string(kappa) + "_lamb=" + to_string(lambda) + "_cs=" + to_string(size[0]) + "bs=" + to_string(batchsize) + "_lr="+to_string(lr) + ".csv");
	//thermalize
	while (true) {
		session->run(feedDic, true, 10);
		if (overall > 500) {
			cd->optimize(10, 1, true,true);
		}
		else {
			cd->optimize(10, 1, true);
		}
		if (runningCounter % 20 == 0) {
			avgK /= 20;
			avgL /= 20;
			std::cout << "\r" << "                                                                           ";
			std::cout << "\r" << "Average over last 20 samples: avK=" << avgK << " and avgL=" << avgL;
			runningCounter = 1;

			if (abs(lastAvgK - avgK) < thresholdK && abs(lastAvgL - avgL) < thresholdL) {
				std::cout << std::endl << "difference smaller than " << thresholdK << "  " << thresholdL << std::endl;
				lastAvgK = avgK;
				lastAvgL = avgL;
				break;
			}
			lastAvgK = avgK;
			lastAvgL = avgL;
			avgK = 0;
			avgL = 0;
		}
		if (*kap->value < 0) {
			kap->value = make_shared<Tensor>(Tensor({ 1 }, { 0 }));
		}
		if (*lamb->value < 0) {
			lamb->value = make_shared<Tensor>(Tensor({ 1 }, { 0 }));
		}

		avgK += *kap->value;
		avgL += *lamb->value;
		fileToSave << *kap->value << "," << *lamb->value << std::endl;
		runningCounter++;

	}
	fileToSave.close();
	std::cout << "Thermalized: kappa=" << lastAvgK << "  lambda=" << lastAvgL << std::endl;

}

void RG2DTest::operator()(string name, map<string, double> num_vars, map<string, string> str_vars, map<string, vector<double>> list_vars)
{
	auto size = getIntVector("chainsize", num_vars, list_vars);
	auto bs = getIntVector("batchsize", num_vars, list_vars);
	auto kappa = getDoubleVector("kappa", num_vars, list_vars);
	auto lamb = getDoubleVector("lambda", num_vars, list_vars);
	double lr = num_vars.find("learningrate") == num_vars.end() ? 0.01 : num_vars["learningrate"];
	for (auto b : bs) {
		for (auto k : kappa) {
			for (auto l : lamb) {
				if (name == "test2dConvergence") {
					test2dConvergence(size, b, k, l, lr);
				}
			}
		}
	}
	
}
