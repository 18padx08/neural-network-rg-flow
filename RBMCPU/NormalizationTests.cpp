#include "NormalizationTests.h"

using namespace ct;

NormalizationTests::NormalizationTests()
{
}


NormalizationTests::~NormalizationTests()
{
}

void NormalizationTests::run()
{
	int chainSize = 2048;
	double kappa = 0.3;
	//we need some MC reference
	Phi1D phi4(chainSize, kappa, 0, 0, 0);
	shared_ptr<Graph> graph = RBMCompTree::getRBMGraph();
	auto k = graph->getVarForName("kappa");
	k->value = make_shared<Tensor>(Tensor({ 1 }, { kappa }));
	Session session(graph);
	std::ofstream output("norm_tests.csv");
	std::ofstream test("fitter.csv");
	while (true) {
		double corrWOscalingMC = 0;
		double corrMC = 0;
		double scorrMC = 0;
		double corrNN = 0;
		double corrNN2 = 0;
		int trials = 500;
		vector<double> randomState(chainSize);
		vector<double> zeros(chainSize);
		std::normal_distribution<double> normal(0, 1);
		std::default_random_engine engine(time(NULL));
		for (int k = 0; k < chainSize; k++) {
			randomState[k] = normal(engine);
			zeros[k] = 0;
		}


		for (int i = 0; i < trials; i++) {
			phi4.fftUpdate();
			double tmpm = (log(abs(phi4.getCorrelationLength(2))) - log(abs(phi4.getCorrelationLength(4))))/2.0;
			double m = sqrt(1.0 / kappa - 2);
			corrWOscalingMC += phi4.getCorrelationLength(2);
			//double scaling = 1.0 / sqrt(abs((tmpCorr / (exp(-m)))));
			scorrMC += phi4.getCorrelationLength(4);
			//phi4.rescaleFields(scaling);
			map<string, shared_ptr<Tensor>> feedDic = { { "x", make_shared<Tensor>(Tensor({ chainSize },randomState)) } };
			
				session.run(feedDic, true, 5);
				auto storageNode = dynamic_pointer_cast<Storage>(graph->storages["visibles_raw"]);
				auto chain = (*storageNode->storage[5]);
				double tmpCorrNN = 0;
				double tmpCorr2NN = 0;
	
				for (int k = 0; k < chain.dimensions[0] / 2; k++) {
					tmpCorrNN += chain[{2 * k}] * chain[{2 * k + 2}] ;
					tmpCorr2NN += chain[{2 * k}] * chain[{2 * k + 4}] ;
				}			
				tmpCorrNN /= chain.dimensions[0] / 2;
				tmpCorr2NN /= chain.dimensions[0] / 2;
			
				corrNN += tmpCorrNN ;
				corrNN2 += tmpCorr2NN;
		}
		corrWOscalingMC /= trials;
		corrMC = corrWOscalingMC;
		scorrMC /= trials;
		corrNN /= trials ;
		corrNN2 /= trials;
		//scale
		double calcM = (log(abs(corrNN)) - log(abs(corrNN2)))/2.0;
		double scale = corrNN / exp(-2 * calcM);
		corrNN /= scale;
		corrNN2 /= scale;
		double th = exp(-2 * sqrt(1.0 / kappa - 2));
		output << th << "," << scorrMC << "," << corrNN <<std::endl;
		std::cout << "mass: " << calcM << std::endl;
		std::cout << "Correlation length unscaled: " << corrWOscalingMC << std::endl;
		std::cout << "MC nnn correlation length" << scorrMC << std::endl;
		std::cout << "Neural Network Correlation Length: " << corrNN << std::endl;
		std::cout << "NNN correlation length" << corrNN2 << std::endl;
		std::cout << "Theoretical value: " << th << std::endl;
		std::cout << "Difference (MC-NN): " << scorrMC - corrNN << "\t[" << (abs(scorrMC - corrNN) < 1.0 / sqrt(trials) ? "PASSED" : "FAILED") << "]" << std::endl;
		std::cout << "Difference (TH-NN): " << th - corrNN << "\t[" << (abs(th - corrNN) < 1.0 / sqrt(trials) ? "PASSED" : "FAILED") << "]" << std::endl;
		std::cout << "Difference (TH-MC): " << th - scorrMC << "\t[" << (abs(th - scorrMC) < 1.0 / sqrt(trials) ? "PASSED" : "FAILED") << "]" << std::endl;
		std::cout << std::endl << std::endl;
		//test.flush();
		//output.flush();
	}
	test.close();
	output.close();

}

void NormalizationTests::runConvTest()
{
	//int chainSize = 1024;
	for (int chainSize = 16; chainSize <= 1024; chainSize *= 2)
	{
		double kappa = 0.48;
		shared_ptr<Graph> graph = RBMCompTree::getRBMGraph();
		auto k = graph->getVarForName("kappa");
		k->value = make_shared<Tensor>(Tensor({ 1 }, { kappa }));
		Session session(graph);

		vector<double> randomState(chainSize);
		vector<double> zeros(chainSize);
		std::normal_distribution<double> normal(0, 1);
		std::default_random_engine engine(time(NULL));
		for (int k = 0; k < chainSize; k++) {
			randomState[k] = normal(engine);
			zeros[k] = 0;
		}
		ofstream theFile("variance_test_" + std::to_string(chainSize) + "_.csv");
		map<string, shared_ptr<Tensor>> feedDic = { { "x", make_shared<Tensor>(Tensor({ chainSize },zeros)) } };
		int count = 0;
		for (int counter = 0; counter < 1000; counter++) {
			session.run(feedDic, true, 1);
			auto storageNode = dynamic_pointer_cast<Storage>(graph->storages["visibles_raw"]);
			auto chain = (*storageNode->storage[1]);
			double tmpCorrNN = 0;
			double tmpCorr2NN = 0;
			for (int i = 0; i < 1; i++) {
				session.run(feedDic, true, 1);
				double tmptmp = 0;
				double tmptmp2 = 0;
				for (int k = 0; k < chain.dimensions[0] / 2; k++) {
					tmptmp += chain[{2 * k}] * chain[{2 * k + 2}];
					tmptmp2 += chain[{2 * k}] * chain[{2 * k + 4}];
				}
				tmpCorrNN += tmptmp / (chain.dimensions[0] / 2);
				tmpCorr2NN += tmptmp2 / (chain.dimensions[0] / 2);
			}
			//	tmpCorrNN /= 100;
				//tmpCorr2NN /= 100;
			feedDic = { { "x", make_shared<Tensor>(chain) } };
			theFile << tmpCorrNN << "," << tmpCorr2NN << std::endl;
			std::cout << "\r" << "                                                                                 ";
			std::cout << "\r" << "m [" << count << "]: " << (log(abs(tmpCorrNN)) - log(abs(tmpCorr2NN))) / 2.0;
			count++;
		}
	}
}

void NormalizationTests::compareLatticeAndNN()
{
}
