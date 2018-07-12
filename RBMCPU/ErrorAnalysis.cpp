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
	int spinChainSize = 2048;
	//we need a ising model
	Phi1D ising(spinChainSize, beta, 0.0, 0, 0);
	//thermalize the ising model
	/*for (int i = 0; i < 25000; i++) {
		ising.monteCarloStep();
	}*/
	double maximum = 0.0;
	double totalCorr = 0;

	std::ofstream error_scatter("data/error_scatter_" + std::to_string(beta)+ "_bs=" + to_string(batchsize) + "_cs=" + to_string(spinChainSize) +".csv");
	std::ofstream responseError("data/response_error" + std::to_string(beta)+ "_bs=" + to_string(batchsize) + "_cs=" + to_string(spinChainSize) + ".csv");
	shared_ptr<Graph> graph = RBMCompTree::getRBMGraph();
	Session session(graph);
	optimizers::ContrastiveDivergence cd(graph, 0.1, 0);
	optimizers::ContrastiveDivergence newCd(graph, 0.05, 0);
	ising.fftUpdate();
	//ising.normalize();
	for (int trial = 0; trial < 200; trial++) {
		//we need a batch
		vector<double> samples(spinChainSize * batchsize);
		//and a dimension
		double corr = 0;
		double secondCorr = 0;
		vector<int> dims = { spinChainSize, 2*batchsize };
		double slope = 0;
		std::ofstream of("data/error_gauss_mc_bj=" + std::to_string(beta) + "_" + std::to_string(trial)+"_bs=" + std::to_string(batchsize) + "_cs="+ std::to_string(spinChainSize)  + ".csv");
		for (int sam = 0; sam < batchsize; sam++) {
			double m = sqrt(1.0 / beta - 2);
			double estm = (log(abs(ising.getCorrelationLength(1))) - log(abs(ising.getCorrelationLength(2))));
			auto scaleFactor = 1.0 / sqrt(ising.getCorrelationLength(1) / exp(-m));
			//std::cout << scaleFactor << std::endl;
			ising.rescaleFields(scaleFactor);
			auto chain = ising.getConfiguration();
			for (int i = 0; i < spinChainSize; i++) {
				samples[i + spinChainSize * sam] = chain[i];
				samples[i + spinChainSize * batchsize] = -chain[i];
			}
			//rescaled ising correlations
			of << ising.getCorrelationLength(1) << "," << ising.getCorrelationLength(2) <<std::endl;
			double tmpSecondCorr = ising.getCorrelationLength(2);
			corr += ising.getCorrelationLength(1);
			secondCorr += tmpSecondCorr;
			
			slope += log(abs(ising.getCorrelationLength(1))) - log(abs(tmpSecondCorr));
			ising.fftUpdate();
			
		}
		corr /= batchsize;
		secondCorr /= batchsize;
		slope /= batchsize;
		
		auto betaj =  corr;
		auto mcError = beta - 1.0/(pow(slope,2)+2);
		auto secondMcError = beta - 1 / (log(betaj) + 2);
		totalCorr += secondCorr;
		//get a rbm comp tree



		//our input node to the network
		map<string, shared_ptr<Tensor>> feedDic;
		//now use this batch to thermalize the network
		//stop thermalization when gradient is flat
		feedDic = { {"x",   make_shared<Tensor>(dims, samples) } };
		
		auto castNode = graph->getVarForName("kappa");
		castNode->value = make_shared<Tensor>(Tensor({ 1 }, { beta }));
		auto Ah = graph->getVarForName("Ah");
		Ah->value = make_shared<Tensor>(Tensor({ 1 }, { 1 }));
		auto Av = graph->getVarForName("Av");
		Av->value = make_shared<Tensor>(Tensor({ 1 }, { 1 }));;
		
		double prev = -1.0;
		double next = -1.0;
		//std::ofstream of("error_test_" + std::to_string(trial) + ".csv");
		int counter = 0;
		int events = 0;
		double gradient = 0;
		double average = 0;
		double lastAverage = 0;
		int overalCounter = 0;
		int loops = 0;
		do {
			if (counter % 50 == 0 && counter != 0) {
				//check if average is smaller
				if (abs(average - lastAverage) < 0.1*abs(average) * (1.0/sqrt(batchsize))) break;
				lastAverage = average;
				average = 0;
				counter = 0;
				loops++;
			}
			prev = *castNode->value;
			session.run(feedDic, true, 3);
			cd.optimize(3, 1.0, true);
			next = *castNode->value;
			if (next < 0) {
				*castNode->value = Tensor({ 1 }, { 0 });
			}
			//of << (double)*castNode->value << std::endl;
			std::cout << "\r" << "                                              ";
			std::cout << "\r kappa=" << (double)*castNode->value << "; Ah="  << (double)*Ah->value << ";Av=" << (double)*Av->value ;
			counter++;
			overalCounter++;
			average += (double)*castNode->value / 50;
		} while (true || overalCounter > 1000);
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
			session.run(feedDic, true, 5);
			newCd.optimize(5, 5, true);
			if (*castNode->value < 0) {
				av += 0;
			}
			else {
				av += abs(*castNode->value);
			}
			//current value for kappa
			newOf << ((double)*castNode->value <0? 0 : (double)*castNode->value) << std::endl;
		}
		newOf.close();
		error_scatter <<beta - av / (batchsize) << ", " << mcError << std::endl;
		std::cout << "network error naive: " << av / (batchsize) - beta << std::endl;
		std::cout << "mc error: " << mcError << std::endl;
		std::cout << std::endl << "network was trained, now check network response" << std::endl;
		castNode->value = make_shared<Tensor>(Tensor({ 1 }, { av / batchsize }));
		//initialize not so random random state
		for (int theBatch = 0; theBatch < batchsize; theBatch++) {
			for (int i = 0; i < spinChainSize; i++) {
				samples[i + theBatch * spinChainSize] = i % 2 == 0 ? 0 : 0;
			}
		}
		std::cout << "Batch initialized, lets Gibbs Sample" << std::endl;
		feedDic.clear();
		feedDic = { { "x", make_shared<Tensor>(dims,samples) } };
		
		std::cout << "Gibbs sampling finished, calculate mean" << std::endl;
		auto storageNode = dynamic_pointer_cast<Storage>(graph->storages["visibles_raw"].lock());
		auto hiddenStorage = dynamic_pointer_cast<Storage>(graph->storages["hiddens_raw"].lock());
		double trainedCorr = 0;
		double trainedHidden = 0;
		session.run(feedDic, true, 100);
		auto chains = (*storageNode->storage[100]);
		auto hiddenChain = (*hiddenStorage->storage[100]);
	
		std::ofstream gauss("data/error_gauss_nn_bj=" + to_string(beta) + "_" + to_string(trial) + "_bs=" + std::to_string(batchsize) + "_cs=" + std::to_string(spinChainSize) + ".csv");
		int counter2 = 0;
		double normalization = 0;
		trainedCorr = 0;
		trainedHidden = 0;
		double trainedCorr2 = 0;
		for (int s = 0; s < batchsize; s++) {
			double tmpCorr = 0;
			double tmpHidden = 0;
			
			for (int i = 0; i < spinChainSize / 2; i++) {
				double nnv = 0;
				double nnh = 0;
				nnv = chains[{2 * i, s, 0}] * chains[{(2 * i + 2), s, 0}];
				nnh = hiddenChain[{i, s, 0}] * hiddenChain[{(i + 1), s, 0}];
			
				tmpCorr += nnv;
				trainedCorr2 += chains[{2 * i, s, 0}] * chains[{(2 * i + 4), s, 0}];
				tmpHidden += nnh;
				trainedCorr += nnv;
				trainedHidden += nnh;
				counter2++;
			}
			
			tmpCorr /= spinChainSize/2.0 ;
			tmpHidden /= spinChainSize/2.0;
			gauss << tmpCorr << "," << tmpHidden << std::endl;
		}
		//session.run(feedDic, true, 100);
		//chains = (*storageNode->storage[100]);
		//hiddenChain = (*hiddenStorage->storage[100]);
		trainedCorr /= counter2;
		trainedHidden /= counter2 ;
		trainedCorr2 /= counter2 ;
		double calcM = (log(abs(trainedCorr)) - log(abs(trainedCorr2))) / 2.0;
		double m = sqrt(1.0 / beta - 2);
		double scale = exp(-2 * calcM) / trainedCorr;
		//trainedCorr *= scale;
		//trainedHidden *= scale;

		//error in visible layer, error in hidden layer, error of monte carlo
		responseError <<  exp(-2*m) - trainedCorr << "," << exp(-2*m) - trainedHidden << "," << mcError << "," << trainedCorr << "," << trainedHidden << ","<<secondCorr << std::endl;
		std::cout << std::endl;
		std::cout << "mass: " << m << " - " << calcM << " = " << m - calcM << std::endl;
		std::cout << "correlation network " << trainedCorr << std::endl;
		std::cout << "correlation mc " << secondCorr << std::endl;
		std::cout << "hidden network" << trainedHidden << std::endl;
		std::cout << "correlation theoretical: " << exp(-2*m) << std::endl;
		std::cout << "theory - visible : " << exp(-2*m) - trainedCorr << " theory - hidden" << exp(-2*m) -  trainedHidden << " theory - mc" << exp(-2*m) - secondCorr <<std::endl;
		std::cout << trainedCorr << " / " << trainedCorr2 << "  = " << trainedCorr / trainedCorr2 << std::endl;
		std::cout << std::endl;
		std::cout << "============" << std::endl;
		responseError.flush();
		error_scatter.flush();
	}
	totalCorr /= 200;
	std::cout << "Final Value for the correlation: " << totalCorr;
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
	auto variable = dynamic_pointer_cast<Variable>(graph->variables[0].lock());
	//set network parameter to theoretical value
	variable->value = make_shared<Tensor>(Tensor({ 1 }, { -2*beta }));
	Session session(graph);


}

void ErrorAnalysis::plotNonZeroLamTests(double kappa, double lambda)
{
	int batchsize = 20;
	int chainsize = 1024;
	Phi1D phi4(chainsize, kappa, lambda, 0, 0);
	phi4.useWolff = true;
	//thermalize since we cannot use the fourier update
	phi4.thermalize();

	//setup the neural network
	auto graph = RBMCompTree::getRBMGraph();
	auto cd = make_shared<ct::optimizers::ContrastiveDivergence>(ct::optimizers::ContrastiveDivergence(graph,0.1));
	auto session = make_shared<Session>(Session(graph));
	auto lambdaNN = graph->getVarForName("lambda");
	auto kappaNN = graph->getVarForName("kappa");
	auto Ah = graph->getVarForName("Ah");
	auto Av = graph->getVarForName("Av");
	lambdaNN->value = make_shared<Tensor>(Tensor({ 1 }, { lambda }));
	kappaNN->value = make_shared<Tensor>(Tensor({ 1 }, { kappa }));
	Ah->value = make_shared<Tensor>(Tensor({ 1 }, { 1 }));
	Av->value = make_shared<Tensor>(Tensor({ 1 }, { 1 }));

	map<string, shared_ptr<Tensor>> feedDic;
	vector<int> dims = { chainsize, batchsize };
	vector<double> samples(batchsize * chainsize);
	for (int trials = 0; trials < 200; trials++) {
		//generate samples
		double secondCorr = 0;
		lambdaNN->value = make_shared<Tensor>(Tensor({ 1 }, { lambda }));
		kappaNN->value = make_shared<Tensor>(Tensor({ 1 }, { kappa }));
		Ah->value = make_shared<Tensor>(Tensor({ 1 }, { 1 }));
		Av->value = make_shared<Tensor>(Tensor({ 1 }, { 1 }));
		for (int i = 0; i < batchsize; i++) {
			phi4.monteCarloSweep();
			auto tmp = phi4.getConfiguration();
			secondCorr += phi4.getCorrelationLength(2);
			for (int j = 0; j < chainsize; j++) {
				samples[j + i * chainsize] = tmp[j];
			}
		}
		secondCorr /= batchsize;
		//setup the feeddic
		feedDic = { { "x", make_shared<Tensor>(dims, samples) } };
		int overalCounter = 0;
		int loops = 0;
		double average = 0;
		int counter = 0;
		double lastAverage = 0;
		do {
			if (counter % 50 == 0 && counter != 0) {
				//check if average is smaller
				if (abs(average - lastAverage) < 0.1*abs(average) * (1.0 / sqrt(batchsize))) break;
				lastAverage = average;
				average = 0;
				counter = 0;
				loops++;
			}
			//prev = *castNode->value;
			session->run(feedDic, true, 3);
			cd->optimize(3, 1.0, true);
			//next = *castNode->value;
			/*if (next < 0) {
				*castNode->value = Tensor({ 1 }, { 0 });
			}*/
			//of << (double)*castNode->value << std::endl;
			std::cout << "\r" << "                                              ";
			std::cout << "\r kappa=" << (double)*kappaNN->value << "; lambda=" << (double)*lambdaNN->value << ", Ah=" << (double)*Ah->value << ", Av="<<(double)*Av->value;
			counter++;
			overalCounter++;
			average += (double)*kappaNN->value / 50;
		} while (true && overalCounter < 100);

		std::cout << std::endl;
		std::cout << "Thermalized" << std::endl;

		//do some measurements
		std::cout << "Batch initialized, lets Gibbs Sample" << std::endl;
		feedDic.clear();
		feedDic = { { "x", make_shared<Tensor>(dims,samples) } };

		std::cout << "Gibbs sampling finished, calculate mean" << std::endl;
		auto storageNode = dynamic_pointer_cast<Storage>(graph->storages["visibles_raw"].lock());
		auto hiddenStorage = dynamic_pointer_cast<Storage>(graph->storages["hiddens_raw"].lock());
		double trainedCorr = 0;
		double trainedHidden = 0;
		session->run(feedDic, true, 100);
		auto chains = (*storageNode->storage[100]);
		auto hiddenChain = (*hiddenStorage->storage[100]);

		int counter2 = 0;
		double normalization = 0;
		trainedCorr = 0;
		trainedHidden = 0;
		double trainedCorr2 = 0;
		double averageSquared = 0;
		double averageV = 0;
		for (int s = 0; s < batchsize; s++) {
			double tmpCorr = 0;
			double tmpHidden = 0;

			for (int i = 0; i < chainsize / 2; i++) {
				double nnv = 0;
				double nnh = 0;
				double tmpA = chains[{2 * i, s, 0}];
				nnv = tmpA * chains[{(2 * i + 2), s, 0}];
				nnh = hiddenChain[{i, s, 0}] * hiddenChain[{(i + 1), s, 0}];
				averageSquared += pow(tmpA,2);
				averageV += tmpA;
				tmpCorr += nnv;
				trainedCorr2 += chains[{2 * i, s, 0}] * chains[{(2 * i + 4), s, 0}];
				tmpHidden += nnh;
				trainedCorr += nnv;
				trainedHidden += nnh;
				counter2++;
				
			}
			

			tmpCorr /= chainsize / 2.0;
			tmpHidden /= chainsize / 2.0;
			//gauss << tmpCorr << "," << tmpHidden << std::endl;
		}
		//session.run(feedDic, true, 100);
		//chains = (*storageNode->storage[100]);
		//hiddenChain = (*hiddenStorage->storage[100]);
		trainedCorr /= counter2;
		trainedHidden /= counter2;
		trainedCorr2 /= counter2;
		averageV /= counter2;
		averageSquared /= counter2;
		double calcM = (log(abs(trainedCorr)) - log(abs(trainedCorr2))) / 2.0;
		double m = sqrt( abs((1.0 - 2.0*lambda) / kappa - 2.0));
		double scale = exp(-2 * calcM) / trainedCorr;
		//trainedCorr *= scale;
		//trainedHidden *= scale;

		//error in visible layer, error in hidden layer, error of monte carlo
		//responseError << trainedCorr << "," << trainedHidden << "," << mcError << "," << trainedCorr << "," << trainedHidden << "," << secondCorr << std::endl;
		std::cout << std::endl;
		std::cout << "average squared: " << averageSquared << " " << phi4.squaredVolumeAverage() << std::endl;
		std::cout << "average: " << averageV << "  " << phi4.volumeAverage() << std::endl;
		std::cout << "mass: " << m << " - " << calcM << " = " << m - calcM << std::endl;
		std::cout << "correlation network " << trainedCorr << std::endl;
		std::cout << "correlation mc " << secondCorr << std::endl;
		std::cout << "hidden network" << trainedHidden << std::endl;
		std::cout << "correlation theoretical: " << exp(-2 * m) << std::endl;
		std::cout << "theory - visible : " << exp(-2 * m) - trainedCorr << " theory - hidden" << exp(-2 * m) - trainedHidden << " theory - mc" << exp(-2 * m) - secondCorr << std::endl;
		std::cout << trainedCorr << " / " << trainedCorr2 << "  = " << trainedCorr / trainedCorr2 << std::endl;
		std::cout << std::endl;
		std::cout << "============" << std::endl;
	}
	

}
