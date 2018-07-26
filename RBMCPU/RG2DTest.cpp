#include "RG2DTest.h"



RG2DTest::RG2DTest()
{
}


RG2DTest::~RG2DTest()
{

}

void RG2DTest::plot2DRGFlow(vector<int> size, int batchsize, int layersize,  double kappa, double lambda, double lr)
{

	int max_iterations = 400;
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

	map<string, shared_ptr<Tensor>> feedDic;
	feedDic = { { "x" , make_shared<Tensor>(dims, samples) } };
	vector<shared_ptr<Graph>> graphs;
	vector<shared_ptr<Session>> sessions;
	vector<shared_ptr<ContrastiveDivergence2D>> cds;
	vector<shared_ptr<ContrastiveDivergence2D>> newCds;
	for (int l = 0; l < layersize; l++) {
		auto tmp = RBMCompTree::getRBM2DGraph();
		auto s = make_shared<Session>(tmp);
		auto c = make_shared<ContrastiveDivergence2D>(tmp, lr);
		auto newc = make_shared<ContrastiveDivergence2D>(tmp, 0.3*lr);
		auto k = tmp->getVarForName("kappa");
		auto lam = tmp->getVarForName("lambda");
		k->value = make_shared<Tensor>(Tensor({ 1 }, { kappa }));
		lam->value = make_shared<Tensor>(Tensor({ 1 }, { lambda }));
		graphs.push_back(tmp);
		sessions.push_back(s);
		cds.push_back(c);
		newCds.push_back(newc);
	}

	//start thermalizing


	double thresholdK = 0.001;
	double thresholdL = 0.001;

	for (int l = 0; l < layersize; l++) {
		//each layer has to be trained
		auto kap = graphs[l]->getVarForName("kappa");
		auto lamb = graphs[l]->getVarForName("lambda");

		int runningCounter = 1;
		int overall = 0;
		double avgK = 0;
		double avgL = 0;
		double lastAvgK = 0;
		double lastAvgL = 0;
		bool isThermalizing = true;
		std::cout << "Layer[" << l << "] of " << layersize << std::endl;
		while (isThermalizing) {
			if (l > 0) {
				//propagate sample to current layer
				for (int newl = 0; newl < l; newl++) {
					sessions[newl]->run(feedDic, true, 10);
					auto hidden = (dynamic_pointer_cast<Storage>(graphs[newl]->storages["hiddens_raw"].lock())->storage[10]);
					//vals->rescale(sqrt((1 - 2 * *var->value**var->value)));
					feedDic = { { "x", make_shared<Tensor>(Tensor(*hidden)) } };
					//std::cout << newl << std::endl;
				}
			}
			sessions[l]->run(feedDic, true, 10);
			cds[l]->optimize(10, 1, true);

			if (runningCounter % 20 == 0) {
				avgK /= 20;
				avgL /= 20;
				std::cout << "\r" << "                                                                           ";
				std::cout << "\r" << "Average over last 20 samples: avK=" << avgK << " and avgL=" << avgL;
				std::cout << "runningCounter = " << runningCounter << std::endl;
				runningCounter = 1;

				if (abs(lastAvgK - avgK) < thresholdK && abs(lastAvgL - avgL) < thresholdL) {
					
					std::cout << std::endl << "difference smaller than " << abs(lastAvgK - avgK) << "  " << abs(lastAvgL - avgL) << " " << avgK << " " << lastAvgK << " ** " <<avgL << "  " << lastAvgL  << std::endl;
					lastAvgK = avgK;
					lastAvgL = avgL;
					kap->value = make_shared<Tensor>(Tensor({ 1 }, { lastAvgK }));
					lamb->value = make_shared<Tensor>(Tensor({ 1 }, { lastAvgL }));
					isThermalizing = false;
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
			
			runningCounter++;
		}
	}

	std::cout << "We are thermalized!" << std::endl;

	std::ofstream kap("2d_rg_flow_comb_kappa=" + std::to_string(kappa) + "_bs=" + std::to_string(batchsize) + "_cs=" + std::to_string(size[0]) + "_withLam=" + to_string(lambda) + ".csv");
	std::ofstream lamb("2d_rg_flow_comb_lambda=" + std::to_string(lambda) + "_bs=" + std::to_string(batchsize) + "_cs=" + std::to_string(size[0]) + "_withKappa=" + to_string(kappa) + ".csv");

	for (int layer = 0; layer < layersize; layer++) {
		for (int it = 0; it < max_iterations; it++) {
			//each iterations needs a new batch
			for (int sam = 0; sam < batchsize; sam++) {
				auto t = phi4.getConfiguration();
				for (int i = 0; i < size[0]; i++) {
					for (int j = 0; j < size[1]; j++) {
						samples[i + j*size[0] + sam * size[0]*size[1]] = t[i][j];
					}
				}
				phi4.monteCarloSweep();
			}
			//propagate the batch through the layer
			if (it % 10 == 0) {
				std::cout << "\r" << "                                                                                                                       ";
				std::cout << "\r" << "Coupling [" << it << " of " << max_iterations << "]: ";
			}

			feedDic = { { "x", make_shared<Tensor>(Tensor(dims, samples)) } };
			auto var = graphs[layer]->getVarForName("kappa");
			auto lam = graphs[layer]->getVarForName("lambda");
			auto renorm = graphs[layer]->getVarForName("Av");
			for (int i = 0; i <= layer; i++) {
				if (i > 0) {
					//if not the first layer take the output from the last layer
					auto vals = (dynamic_pointer_cast<Storage>(graphs[i - 1]->storages["hiddens_raw"].lock())->storage[5]);
					//vals->rescale(sqrt((1 - 2 * *var->value**var->value)));
					feedDic = { { "x", make_shared<Tensor>(*vals) } };
				}
				sessions[i]->run(feedDic, true, 5);
				if (i == layer) {
					newCds[layer]->optimize(5, 10, true);
				}
			}
			//dynamic_pointer_cast<Variable>(graphList[layer]->variables[0]);
			if (*var->value < 0) {
				*var->value = Tensor({ 1 }, { 0 });
			}
			if (*lam->value < 0) {
				*lam->value = Tensor({ 1 }, { 0 });
			}
			if (it % 10 == 0) {
				std::cout << " kappa " << (double)*(var->value) << " " << (double)*(lam->value) << " ";
			}
			if (it > 200) {
				kap << (double) *(var->value) << ",";
				lamb << (double) *(lam->value) << ",";
			}
		}
		kap << std::endl;
		lamb << std::endl;
	}
	kap.close();
	lamb.close();

}

void RG2DTest::test2dConvergence(vector<int> size, int batchsize, double kappa, double lambda, double lr)
{
	Phi2D phi4(size, kappa, lambda);
	phi4.useWolff = true;
	phi4.thermalize(1000);

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
	double overallOptimK = 0;
	double overallOptimL = 0;
	while (true) {
		session->run(feedDic, true, 10);
		
			auto vals = cd->optimize(10, 1, true);
			overallOptimK += vals[0];
			overallOptimL += vals[1];
			std::cout << vals[0] << "  " << vals[1] << std::endl;
		if (runningCounter % 20 == 0) {
			overallOptimK /= 20;
			overallOptimL /= 20;
			if (abs(overallOptimL) > 0.1*cd->learningRateL) {
				cd->learningRateL += 0.1 * cd->learningRateL;
			}
			else if(abs(overallOptimL) < 0.1*cd->learningRateL){
				//std::cout << abs(overallOptimL) << "  " << cd->learningRateL << std::endl;
				cd->learningRateL -= 0.1 *cd->learningRateL;
			}
			if (abs(overallOptimK) > 0.1*cd->learningRateK) {
				cd->learningRateK += 0.1 * cd->learningRateK;
			}
			else if(abs(overallOptimK) < 0.1*cd->learningRateK)  {
				//std::cout << abs(overallOptimK) << "  " << cd->learningRateK << std::endl;
				cd->learningRateK -= 0.1 *cd->learningRateK;
			}
			avgK /= 20;
			avgL /= 20;
			std::cout << "\r" << "                                                                           ";
			std::cout << "\r" << "Average over last 20 samples: avK=" << avgK << "(" << cd->learningRateK << ") and avgL=" << avgL << "(" << cd->learningRateL << ")";
			runningCounter = 1;

			if (abs(lastAvgK - avgK) < thresholdK && abs(lastAvgL - avgL) < thresholdL) {
				std::cout << std::endl << "difference smaller than " << thresholdK << "  " << thresholdL << std::endl;
				std::cout << "lrL: " << cd->learningRateL << "  lrK: " << cd->learningRateK << std::endl;
				lastAvgK = avgK;
				lastAvgL = avgL;
				break;
			}
			lastAvgK = avgK;
			lastAvgL = avgL;
			avgK = 0;
			avgL = 0;
			overallOptimK = 0;
			overallOptimL = 0;
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
	int layersize = num_vars.find("layersize") == num_vars.end() ? 3 : (int)num_vars["layersize"];
	double lr = num_vars.find("learningrate") == num_vars.end() ? 0.01 : num_vars["learningrate"];
	for (auto b : bs) {
		for (auto k : kappa) {
			for (auto l : lamb) {
				if (name == "test2dConvergence") {
					test2dConvergence(size, b, k, l, lr);
				}
				if (name == "plot2DRGFlow") {
					plot2DRGFlow(size, b, layersize, k, l, lr);
				}
			}
		}
	}
	
}
