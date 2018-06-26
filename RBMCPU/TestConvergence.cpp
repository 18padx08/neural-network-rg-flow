#include "TestConvergence.h"



TestConvergence::TestConvergence()
{
}


TestConvergence::~TestConvergence()
{
}

void TestConvergence::testConvergence(double learningRate, double kappa, double lambda, int chainsize, int batchsize, bool useZ2)
{
	//NN setup
	auto graph = ct::RBMCompTree::getRBMGraph();
	auto cd = make_shared<ct::optimizers::ContrastiveDivergence>(graph, learningRate, 0);
	auto session = make_shared<ct::Session>(graph);
	//MonteCarlo setup
	Phi1D phi(chainsize, kappa, lambda, 0, 0);
	phi.useWolff = true;
	phi.thermalize();
	vector<double> samples(chainsize*2*batchsize);
	ofstream output("conv_test_kappa=" + to_string(kappa) + "_lambda=" + to_string(lambda) + "_cs=" + to_string(chainsize) + "_bs=" + to_string(batchsize) + ".csv");

	double mcCorrelation = 0;
	//make samples
	for (int b = 0; b < batchsize; b++) {
		phi.monteCarloSweep();
		mcCorrelation += phi.getCorrelationLength(2);
		auto config = phi.getConfiguration();
		for (int c = 0; c < config.size(); c++) {
			samples[c + b * chainsize] = config[c];
			samples[c + b * chainsize + batchsize * chainsize] = -config[c];
		}
	}
	auto kap = graph->getVarForName("kappa");
	auto lam = graph->getVarForName("lambda");
	kap->value = make_shared<ct::Tensor>(ct::Tensor({ 1 }, { kappa /2 }));
	lam->value = make_shared<ct::Tensor>(ct::Tensor({ 1 }, { lambda/2 * 1.8}));
	mcCorrelation /= batchsize;
	vector<int> dims = { chainsize,2*batchsize };
	map<string, shared_ptr<ct::Tensor>> feedDic;
	feedDic = { {"x", make_shared<ct::Tensor>(dims, samples)} };
	for (int i = 0; i < 1500; i++) {

		session->run(feedDic, true, 5);
		cd->optimize(5, 0, true);
		if (*lam->value < 0.0) {
			lam->value = make_shared<ct::Tensor>(ct::Tensor({ 1 }, { 0 }));
		}
		if (*kap->value < 0.0) {
			kap->value = make_shared<ct::Tensor>(ct::Tensor({ 1 }, { 0 }));
		}
		auto storageNode = *(*dynamic_pointer_cast<ct::Storage>(graph->storages["visibles_raw"].lock())).storage[5];
		
		double tmpCorr = 0;
		for (int batch = 0; batch < storageNode.dimensions[1]; batch++) {
			for (int site = 0; site < storageNode.dimensions[0]/2; site++) {
				tmpCorr += storageNode[{2*site, batch}] * storageNode[{2*site + 2, batch}];
			}
		}
		tmpCorr /= (storageNode.dimensions[0]/2.0) * storageNode.dimensions[1];
		output << (double)*kap->value << "," << (double)*lam->value << "," << tmpCorr << "," << mcCorrelation << std::endl;
		std::cout << "\r" << "                                                                    ";
		std::cout <<"\r" << "[" << i << "] "<< (double)*kap->value << "  " << (double)*lam->value  << "  " << tmpCorr << " - " << mcCorrelation << " = " << tmpCorr - mcCorrelation;
	}
	output.close();

}

void TestConvergence::operator()(string name, map<string, double> num_vars, map<string, string> str_vars, map<string, vector<double>> list_vars)
{
	auto kappa = this->getDoubleVector("kappa", num_vars,list_vars);
	auto lambda = this->getDoubleVector("lambda", num_vars, list_vars);
	auto chainsize = this->getIntVector("chainsize", num_vars, list_vars);
	auto batchsize = this->getIntVector("batchsize", num_vars, list_vars);
	auto lr = num_vars.find("learningRate") == num_vars.end() ? 0.1 : num_vars["learningRate"];
	if (name == "testConvergence") {
		for (auto k : kappa) {
			for (auto l : lambda) {
				for (auto c : chainsize) {
					for (auto bs : batchsize) {
						testConvergence(lr,k, l, c, bs, false);
					}
				}
			}
		}
	}
}
