#include "ErrorAnalysis.h"



double ErrorAnalysis::errorMonteCarlo(shared_ptr<Tensor> samples)
{
	return 0.0;
}

ErrorAnalysis::ErrorAnalysis()
{
}


ErrorAnalysis::~ErrorAnalysis()
{
}

void ErrorAnalysis::plotErrorOnTraining(double beta)
{
	//batchsize
	int batchsize = 10;
	int spinChainSize = 512;
	//we need a ising model
	Ising1D ising(spinChainSize, beta, 1.0);
	//thermalize the ising model
	for (int i = 0; i < 25000; i++) {
		ising.monteCarloStep();
	}
	std::ofstream error_scatter("error_scatter.csv");
	std::ofstream responseError("response_error.csv");
	shared_ptr<Graph> graph = RBMCompTree::getRBMGraph();
	Session session(graph);
	optimizers::ContrastiveDivergence cd(graph, 0.1, 0);
	optimizers::ContrastiveDivergence newCd(graph, 0.08, 0);

	for (int trial = 0; trial < 200; trial++) {
		//we need a batch
		vector<double> samples(spinChainSize * batchsize);
		//and a dimension
		double corr = 0;
		double secondCorr = 0;
		vector<int> dims = { spinChainSize, batchsize };
		for (int sam = 0; sam < batchsize; sam++) {
			auto t = ising.getConfiguration();
			for (int i = 0; i < spinChainSize; i++) {
				samples[i + sam * spinChainSize] = t[i] <= 0 ? -1 : 1;
				corr += (t[i] <= 0 ? -1 : 1) * (t[(i + 1) % spinChainSize] <= 0 ? -1 : 1);
				secondCorr += (t[i] <= 0 ? -1 : 1) * (t[(i + 2) % spinChainSize] <= 0 ? -1 : 1);
			}
			for (int i = 0; i < 3000; i++) {
				ising.monteCarloStep();
			}
		}
		corr /= batchsize * spinChainSize;
		secondCorr /= batchsize * spinChainSize;
		auto betaj = atanh(corr);
		auto mcError = betaj - beta;

		//get a rbm comp tree



		//our input node to the network
		map<string, shared_ptr<Tensor>> feedDic;
		//now use this batch to thermalize the network
		//stop thermalization when gradient is flat
		feedDic = { {"x",   make_shared<Tensor>(dims, samples) } };
		auto castNode = dynamic_pointer_cast<Variable>(graph->variables[0]);
		castNode->value = make_shared<Tensor>(Tensor({ 1 }, { -2.0 }));
		double prev = -1.0;
		double next = -1.0;
		std::ofstream of("error_test_" + std::to_string(trial) + ".csv");
		int counter = 0;
		int events = 0;
		double gradient = 0;
		double average = 0;
		double lastAverage = 0;
		int loops = 0;
		do {
			if (counter % 50 == 0 && counter != 0) {
				//check if average is smaller
				if (abs(average - lastAverage) < 0.0001) break;
				lastAverage = average;
				average = 0;
				counter = 0;
				loops++;
			}
			prev = *castNode->value;
			session.run(feedDic, true, 10);
			cd.optimize(10, 1.0, false);
			next = *castNode->value;
			of << (double)*castNode->value << std::endl;
			std::cout << "\r" << "                                              ";
			std::cout << "\r" << (double)*castNode->value / 2.0;
			counter++;

			average = (double)*castNode->value / 2.0 / counter;
		} while (true);
		of.close();
		std::cout << std::endl << "============ " << trial << " ==========" << std::endl;
		std::cout << "thermalized after " << counter * loops << " steps: " << (double)*castNode->value << std::endl;
		std::cout << "do some measurements" << std::endl;
		std::ofstream newOf("error_gauss_" + to_string(trial) + ".csv");
		double av = 0;
		for (int batch = 0; batch < batchsize; batch++) {
			session.run(feedDic, true, 100);
			newCd.optimize(100, 1, true);
			av += abs(*castNode->value);
			newOf << (double)*castNode->value << std::endl;
		}
		newOf.close();
		error_scatter <<av / batchsize / 2.0 - beta << ", " << mcError << std::endl;
		std::cout << "network error naive: " << av / batchsize / 2.0 - beta << std::endl;
		std::cout << "mc error: " << mcError << std::endl;
		std::cout << std::endl << "network was trained, now check network response" << std::endl;

		//initialize not so random random state
		for (int theBatch = 0; theBatch < batchsize; theBatch++) {
			for (int i = 0; i < spinChainSize; i++) {
				samples[i + theBatch * spinChainSize] = i % 2 == 0 ? -1 : 1;
			}
		}
		std::cout << "Batch initialized, lets Gibbs Sample" << std::endl;
		feedDic.clear();
		feedDic = { { "x", make_shared<Tensor>(dims,samples) } };
		session.run(feedDic, true, 100);
		std::cout << "Gibbs sampling finished, calculate mean" << std::endl;
		auto storageNode = dynamic_pointer_cast<Storage>(graph->storages["visibles_pooled"]);
		auto hiddenStorage = dynamic_pointer_cast<Storage>(graph->storages["hiddens_pooled"]);
		double trainedCorr = 0;
		double trainedHidden = 0;
		auto chains = (*storageNode->storage[100]);
		auto hiddenChain = (*hiddenStorage->storage[100]);
#pragma omp parallel for reduction(+:trainedCorr, trainedHidden)
		for (int s = 0; s < batchsize; s++) {
#pragma omp parallel for reduction(+:trainedCorr,trainedHidden)
			for (int i = 0; i < spinChainSize; i+=2) {
				trainedCorr += chains[{i, s, 0}] * chains[{i+2%spinChainSize,s,0}];
				trainedHidden += hiddenChain[{i, s, 0}] * hiddenChain[{i + 1 % (spinChainSize/2), s, 0}];
			}
		}
		
		trainedCorr /= batchsize * (spinChainSize/2.0);
		trainedHidden /= batchsize * (spinChainSize / 2.0);
		responseError << -pow(tanh(beta), 2) + trainedCorr << "," << -pow(tanh(beta), 2) + secondCorr << "," << trainedHidden -pow(tanh(beta), 2) << std::endl;
		std::cout << std::endl;
		std::cout << "correlation network " << trainedCorr << std::endl;
		std::cout << "correlation mc " << secondCorr << std::endl;
		std::cout << "hidden network" << trainedHidden << std::endl;
		std::cout << "theoretical value " << pow(tanh(beta), 2) << " mc error: " << -pow(tanh(beta),2) + secondCorr << " nn error " << -pow(tanh(beta), 2) + trainedCorr <<std::endl;
		std::cout << std::endl;
		std::cout << "============" << std::endl;
		responseError.flush();
		error_scatter.flush();
	}
	error_scatter.close();
}

void ErrorAnalysis::plotErrorOfResponse(double beta)
{
	int batchsize = 50;
	int spinChainSize = 512;
	//we need a ising model
	Ising1D ising(spinChainSize, beta, 1.0);

	//we need a (trained) network
	//get a rbm comp tree
	shared_ptr<Graph> graph = RBMCompTree::getRBMGraph();
	auto variable = dynamic_pointer_cast<Variable>(graph->variables[0]);
	//set network parameter to theoretical value
	variable->value = make_shared<Tensor>(Tensor({ 1 }, { -2*beta }));
	Session session(graph);


}
