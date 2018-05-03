#include "ErrorAnalysis.h"
#include "Phi1D.h"


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

void ErrorAnalysis::plotErrorOnTraining(double beta, int bs)
{
	//batchsize
	int batchsize = bs;
	int spinChainSize = 512;
	//we need a ising model
	Phi1D ising(spinChainSize, beta, 0.0, 0, beta);
	//thermalize the ising model
	/*for (int i = 0; i < 25000; i++) {
		ising.monteCarloStep();
	}*/
	double maximum = 0.0;
	ising.useWolff = true;
	for (int i = 0; i < 500; i++) {
		ising.monteCarloSweep();
	}
	std::ofstream error_scatter("data/error_scatter_" + std::to_string(beta)+ "_bs=" + to_string(batchsize) + "_cs=" + to_string(spinChainSize) +".csv");
	std::ofstream responseError("data/response_error" + std::to_string(beta)+ "_bs=" + to_string(batchsize) + "_cs=" + to_string(spinChainSize) + ".csv");
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
		std::ofstream of("data/error_gauss_mc_bj=" + std::to_string(beta) + "_" + std::to_string(trial)+"_bs=" + std::to_string(batchsize) + "_cs="+ std::to_string(spinChainSize)  + ".csv");
		for (int sam = 0; sam < batchsize; sam++) {
			auto t = ising.getConfiguration();
			maximum += ising.getMax();
			double tmpCorr = 0;
			double tmpSecondCorr = 0;
			for (int i = 0; i < spinChainSize; i++) {
				samples[i + sam * spinChainSize] = t[i] ;
				tmpCorr += (t[i]) * (t[(i + 1) % spinChainSize] );
				tmpSecondCorr += (t[i] ) * (t[(i + 2) % spinChainSize] );
				corr += (t[i] ) * (t[(i + 1) % spinChainSize] );
				secondCorr += (t[i] ) * (t[(i + 2) % spinChainSize] );
				
			}
			of << (double)tmpSecondCorr/spinChainSize << std::endl;
			/*for (int i = 0; i < 3000; i++) {
				ising.monteCarloStep();
			}*/
			ising.monteCarloSweep();
		}
		maximum /= batchsize;
		corr /= batchsize * spinChainSize /maximum;
		secondCorr /= batchsize * spinChainSize /maximum;
		auto betaj = atanh(corr);
		auto mcError = betaj - beta;

		//get a rbm comp tree



		//our input node to the network
		map<string, shared_ptr<Tensor>> feedDic;
		//now use this batch to thermalize the network
		//stop thermalization when gradient is flat
		feedDic = { {"x",   make_shared<Tensor>(dims, samples) } };
		auto castNode = dynamic_pointer_cast<Variable>(graph->variables[0]);
		if (castNode->name != "A") {
			castNode = dynamic_pointer_cast<Variable>(graph->variables[1]);
		}
		castNode->value = make_shared<Tensor>(Tensor({ 1 }, { -beta*2 }));
		auto scaling = dynamic_pointer_cast<Variable>(graph->variables[1]);
		if (scaling->name != "s") {
			scaling = dynamic_pointer_cast<Variable>(graph->variables[0]);
		}
		scaling->value = make_shared<Tensor>(Tensor({ 1 }, { 1.0 }));
		double prev = -1.0;
		double next = -1.0;
		//std::ofstream of("error_test_" + std::to_string(trial) + ".csv");
		int counter = 0;
		int events = 0;
		double gradient = 0;
		double average = 0;
		double lastAverage = 0;
		int loops = 0;
		do {
			if (counter % 50 == 0 && counter != 0) {
				//check if average is smaller
				if (abs(average - lastAverage) < 0.01*abs(average) * (1.0/sqrt(batchsize))) break;
				lastAverage = average;
				average = 0;
				counter = 0;
				loops++;
			}
			prev = *castNode->value;
			session.run(feedDic, true, 10);
			cd.optimize(10, 1.0, true);
			next = *castNode->value;
			//of << (double)*castNode->value << std::endl;
			std::cout << "\r" << "                                              ";
			std::cout << "\r" << (double)*castNode->value / 2.0 << "   "  << (double)*scaling->value / 2.0;
			counter++;

			average += (double)*castNode->value / 2.0 / 50;
		} while (true);
		//of.close();
		/*for (int i = 0; i < 1000; i++) {
			session.run(feedDic, true, 1);
			//cd.optimize(1, 1.0, true);
			std::cout << "\r" << "                                              ";
			std::cout << "\r" << (double)*castNode->value / 2.0 << "   " << (double)*scaling->value / 2.0;
		}*/
		std::cout << std::endl << "============ " << trial << " ==========" << std::endl;
		std::cout << "thermalized after " << counter * loops << " steps: " << (double)*castNode->value << std::endl;
		std::cout << "do some measurements" << std::endl;
		std::ofstream newOf("data/error_gauss_bj=" + to_string(beta) + "_" + to_string(trial) + "_bs=" + std::to_string(batchsize) + "_cs="+ std::to_string(spinChainSize) + ".csv");
		double av = 0;
		for (int batch = 0; batch < batchsize; batch++) {
			session.run(feedDic, true, 10);
			newCd.optimize(10, 5, true);
			av += abs(*castNode->value);
			newOf << (double)*castNode->value << std::endl;
		}
		newOf.close();
		error_scatter <<av / (batchsize) / 2.0 - beta << ", " << mcError << std::endl;
		std::cout << "network error naive: " << av / (batchsize) / 2.0 - beta << std::endl;
		std::cout << "mc error: " << mcError << std::endl;
		std::cout << std::endl << "network was trained, now check network response" << std::endl;
		castNode->value = make_shared<Tensor>(Tensor({ 1 }, { av / batchsize }));
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
		auto storageNode = dynamic_pointer_cast<Storage>(graph->storages["visibles_raw"]);
		auto hiddenStorage = dynamic_pointer_cast<Storage>(graph->storages["hiddens_raw"]);
		double trainedCorr = 0;
		double trainedHidden = 0;
		auto chains = (*storageNode->storage[100]);
		auto hiddenChain = (*hiddenStorage->storage[100]);
		std::ofstream gauss("data/error_gauss_nn_bj=" + to_string(beta) + "_" + to_string(trial) + "_bs=" + std::to_string(batchsize) + "_cs=" + std::to_string(spinChainSize) + ".csv");
		int counter2 = 0;
		for (int s = 0; s < batchsize; s++) {
			double tmpCorr = 0;
			double tmpHidden = 0;
			for (int i = 0; i < spinChainSize/2; i++) {
				tmpCorr+= chains[{2*i, s, 0}] * chains[{(2*i + 2) % spinChainSize, s, 0}];
				tmpHidden += hiddenChain[{i, s, 0}] * hiddenChain[{(i + 1) % (spinChainSize / 2), s, 0}];
				trainedCorr += chains[{2*i, s, 0}] * chains[{(2*i+2)%spinChainSize,s,0}];
				//std::cout << hiddenChain[{i, s, 0}] << std::endl;
				trainedHidden += hiddenChain[{i, s, 0}] * hiddenChain[{(i + 1) % (spinChainSize/2), s, 0}];
				counter2++;
			}
			gauss << tmpCorr /( spinChainSize / 2.0) << "," << tmpHidden / (spinChainSize / 2.0) << std::endl;
		}
		
		trainedCorr /= counter2;
		trainedHidden /= counter2;
		responseError << -pow(tanh(beta), 2) + trainedCorr << "," << -pow(tanh(beta), 2) + secondCorr << "," << trainedHidden -pow(tanh(beta), 2) << std::endl;
		std::cout << std::endl;
		std::cout << "correlation network " << trainedCorr << std::endl;
		std::cout << "correlation mc " << secondCorr << std::endl;
		std::cout << "hidden network" << trainedHidden << std::endl;
		std::cout << " mc error - visible : " << -trainedCorr + secondCorr << " mc - hidden " <<  - trainedHidden + secondCorr <<std::endl;
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
