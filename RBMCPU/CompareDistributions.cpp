#include "CompareDistributions.h"



CompareDistributions::CompareDistributions()
{
}


CompareDistributions::~CompareDistributions()
{
}

void CompareDistributions::runTest(double kappa, double lambda, double lr, int chainsize, int batchsize)
{
	auto graph = ct::RBMCompTree::getRBMGraph();
	auto cd = make_shared<ct::optimizers::ContrastiveDivergence>(graph, lr, 0);
	auto session = make_shared<ct::Session>(graph);
	//MonteCarlo setup
	Phi1D phi(chainsize, kappa, lambda, 0, 0);
	phi.useWolff = true;
	phi.thermalize();
	phi.thermalize();
	vector<double> samples(chainsize * 2 * batchsize);
	double m = 0;
	double fakem = 0;
	ofstream output("compare_dists_to_mc_lr=" + to_string(lr) + "_kappa=" + to_string(kappa) + "_lambda=" + to_string(lambda) + "_cs=" + to_string(chainsize) + "_bs=" + to_string(batchsize) + ".csv");
	for (int b = 0; b < batchsize; b++) {
		phi.monteCarloSweep();
		m += log(abs(phi.getCorrelationLength(1))) - log(abs(phi.getCorrelationLength(2)));
		fakem += abs(phi.getCorrelationLength(1));
		auto config = phi.getConfiguration();
		for (int c = 0; c < config.size(); c++) {
			output << config[c] << "," << -config[c] << ",";
			samples[c + b * chainsize] = config[c];
			samples[c + b * chainsize + batchsize * chainsize] = -config[c];
		}
	}
	m /= batchsize;
	std::cout << "real mass: " << m << std::endl;
	m = exp(-sqrt((1.0 / (kappa)-2)));
	fakem /= batchsize;
	fakem = exp(-fakem);
	std::cout << "Mass: " << m << "  " << fakem << std::endl;
	output << std::endl;
	map<string, shared_ptr<ct::Tensor>> feedDic;
	vector<int> dims = { chainsize,2 * batchsize };

	feedDic = { { "x", make_shared<ct::Tensor>(dims, samples) } };
	auto kap = graph->getVarForName("kappa");
	auto lam = graph->getVarForName("lambda");
	kap->value = make_shared<ct::Tensor>(ct::Tensor({ 1 }, { kappa }));
	lam->value = make_shared<ct::Tensor>(ct::Tensor({ 1 }, { lambda }));

	for (int i = 0; i < 500; i++) {

		session->run(feedDic, true, 5);
		cd->optimize(5, 0, true);
		if (*lam->value < 0.0) {
			lam->value = make_shared<ct::Tensor>(ct::Tensor({ 1 }, { 0 }));
		}
		if (*kap->value < 0.0) {
			kap->value = make_shared<ct::Tensor>(ct::Tensor({ 1 }, { 0 }));
		}
		//auto storageNode = *(*dynamic_pointer_cast<ct::Storage>(graph->storages["visibles_raw"].lock())).storage[5];

		std::cout << "\r" << "                                                                    ";
		std::cout << "\r" << "[" << i << "] " << (double)*kap->value << "  " << (double)*lam->value;
	}
	//thermalized
	session->run(feedDic, true, 5);
	auto visibles = *(*dynamic_pointer_cast<ct::Storage>(graph->storages["visibles_raw"].lock())).storage[5];
	auto hiddens = *(*dynamic_pointer_cast<ct::Storage>(graph->storages["hiddens_raw"].lock())).storage[5];
	for (int b = 0; b < hiddens.dimensions[1]; b++) {
		for (int i = 0; i < hiddens.dimensions[0]; i++) {
			output << hiddens[{i, b}] << "," << -hiddens[{i, b}] << "," << visibles[{2 * i, b}] << "," << -visibles[{2 * i, b}] << ",";
		}
	}
	output.close();
}

void CompareDistributions::operator()(string name, map<string, double> num_vars, map<string, string> str_vars, map<string, vector<double>> list_vars)
{
	auto kappa = this->getDoubleVector("kappa", num_vars, list_vars);
	auto lambda = this->getDoubleVector("lambda", num_vars, list_vars);
	auto chainsize = this->getIntVector("chainsize", num_vars, list_vars);
	auto batchsize = this->getIntVector("batchsize", num_vars, list_vars);
	auto lr = num_vars.find("learningRate") == num_vars.end() ? 0.1 : num_vars["learningRate"];
	for (auto k : kappa) {
		for (auto l : lambda) {
			for (auto c : chainsize) {
				for (auto bs : batchsize) {
					if (name == "compareDistribution") {
						runTest(k, l,lr, c, bs);
					}
				}
			}
		}
	}
}
