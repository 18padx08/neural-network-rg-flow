#include "TestLoop.h"
#include "Phi1D.h"
#include <iostream>
#include <fstream>

TestLoop::TestLoop()
{
}


TestLoop::~TestLoop()
{
}

void TestLoop::run()
{
	double kappa = 0.2;
	int spinChainSize = 128;
	int batchsize = 1;
	Phi1D phi4(spinChainSize,kappa,0,0,0);
	shared_ptr<Graph> graph = RBMCompTree::getRBMGraph();
	Session session(graph);
	*graph->getVarForName("kappa")->value = Tensor({1}, {kappa});
	*graph->getVarForName("Av")->value = Tensor({ 1 }, { 1 });
	*graph->getVarForName("Ah")->value = Tensor({ 1 }, { 1 });
	std::ofstream of("testloop.csv");
	vector<double> samples(spinChainSize * batchsize);
	auto storageNode = dynamic_pointer_cast<Storage>(graph->storages["visibles_raw"]);
	auto hiddenStorage = dynamic_pointer_cast<Storage>(graph->storages["hiddens_raw"]);
	for (int i = 0; i < 5000; i++) {
		std::cout << "\r                                                                     ";
		std::cout << "\rStep " << i << " of" << " 500";
		phi4.fftUpdate();
		for (int theBatch = 0; theBatch < batchsize; theBatch++) {
			for (int i = 0; i < spinChainSize; i++) {
				samples[i + theBatch * spinChainSize] = i % 2 == 0 ? 0 : 0;
			}
		}
		
		vector<int> dims = { spinChainSize, batchsize };
		std::map<std::string, std::shared_ptr<Tensor>> feedDic = { { "x", make_shared<Tensor>(dims,samples) } };
		session.run(feedDic, true, 100);
		
		double trainedCorr = 0;
		double trainedHidden = 0;
		auto chains = (*storageNode->storage[100]);
		auto hiddenChain = (*hiddenStorage->storage[100]);
		for (int j = 0; j < hiddenChain.dimensions[0]; j++) {
			trainedCorr += chains[{2 * j}] * chains[{2 * j + 2}];
			trainedHidden += hiddenChain[{j}] * hiddenChain[{j + 1}];
		}
		trainedCorr /= hiddenChain.dimensions[0];
		trainedHidden /= hiddenChain.dimensions[0];
		of << phi4.getCorrelationLength(2) << "," << trainedCorr << "," << trainedHidden << std::endl;
	}
	of.close();
}

void TestLoop::runLoop()
{
	double kappa = 0.4;
	int spinChainSize = 256;
	int batchsize = 1;
	Phi1D phi4(spinChainSize, kappa, 0, 0, 0);
	shared_ptr<Graph> graph = RBMCompTree::getRBMGraph();
	Session session(graph);
	*graph->getVarForName("kappa")->value = Tensor({ 1 }, { kappa });
	*graph->getVarForName("Av")->value = Tensor({ 1 }, { 1 });
	*graph->getVarForName("Ah")->value = Tensor({ 1 }, { 1 });
	vector<double> samples(spinChainSize * batchsize);
	auto storageNode = dynamic_pointer_cast<Storage>(graph->storages["visibles_raw"]);
	auto hiddenStorage = dynamic_pointer_cast<Storage>(graph->storages["hiddens_raw"]);

	phi4.fftUpdate();
	for (int theBatch = 0; theBatch < batchsize; theBatch++) {
		auto conf = phi4.getConfiguration();
		for (int i = 0; i < spinChainSize; i++) {
			samples[i + theBatch * spinChainSize] = i % 2 == 0 ? 2 : 2;
		}
	}

	vector<int> dims = { spinChainSize, batchsize };
	std::map<std::string, std::shared_ptr<Tensor>> feedDic = { { "x", make_shared<Tensor>(dims,samples) } };
	for (int i = 0; i < 100; i++) {
		session.run(feedDic, true, 100);
		double trainedCorr = 0;
		double trainedHidden = 0;
		auto chains = (*storageNode->storage[100]);
		auto hiddenChain = (*hiddenStorage->storage[100]);
		for (int j = 0; j < hiddenChain.dimensions[0]; j++) {
			samples[2 * j] = chains[{2 * j}];
			trainedCorr += chains[{2 * j}] * chains[{2 * j + 2}];
			trainedHidden += hiddenChain[{j}] * hiddenChain[{j + 1}];
		}
		feedDic = { {"x", make_shared<Tensor>(dims, samples)} };
		trainedCorr /= hiddenChain.dimensions[0];
		trainedHidden /= hiddenChain.dimensions[0];
		phi4.fftUpdate();
		auto conf = phi4.getConfiguration();


		std::cout << "\r                                                                                               ";
		std::cout << "\rCorrelation (MC): " << phi4.getCorrelationLength(2) << "; Correlation (vNN): " << trainedCorr << "; Correlation (hNN): " << trainedHidden;
	}
}
