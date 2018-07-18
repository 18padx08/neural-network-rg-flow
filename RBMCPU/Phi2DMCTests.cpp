#include "Phi2DMCTests.h"



Phi2DMCTests::Phi2DMCTests()
{
}


Phi2DMCTests::~Phi2DMCTests()
{
}

void Phi2DMCTests::criticalLineTest(vector<int> chainsize, vector<double> kappas, vector<double> lambdas)
{
	
	for (int i = 0; i < lambdas.size(); i++) {
		double l = lambdas[i];
		double k = kappas[0];
		int counter = 0;
		ofstream output("phi2dmctest_lambda=" + to_string(l) + "_" + to_string(chainsize[0])+".csv");
		//take first value as base
		double baseValue = 0.1;
		bool first = true;
		Phi2D phi(chainsize, k, l);
		while(true) {
			
			phi.useWolff = true;
			phi.thermalize();
			
			double absAvg = 0;
			double phi4 = 0;
			double phi2 = 0;
			double vev = 0;
			for (int i = 0; i < 100; i++) {
				phi4 += phi.quarticVolumeAverage();
				phi2 += phi.squaredVolumeAverage();
				absAvg += phi.absoluteVolumeAverage();
				vev += abs(phi.volumeAverage());
				phi.monteCarloSweep();
			}
			vev /= 100;
			absAvg /= 100;
			phi4 /= 100;
			phi2 /= 100;
			if (first)
			{
				baseValue = absAvg*10;
				first = false;

			}
			output << l << "," << k << "," << phi4/(phi2*phi2) <<  "," <<  phi2 - (vev*vev) << std::endl;
			std::cout << "At l=" << l << " k=" << k << " with B_3=" << phi4 / (phi2*phi2) << "   " << phi2 - (vev*vev) << std::endl;
			
			if (k>0.30) {
				counter++;
				if (counter > 10) break;
			}
			k += 0.1;
			
		}
	}
}

void Phi2DMCTests::criticalLineTestNN(vector<int> chainsize, vector<double> kappas, vector<double> lambdas)
{
	auto graph = RBMCompTree::getRBM2DGraph();
	auto kappa = graph->getVarForName("kappa");
	auto lambda = graph->getVarForName("lambda");
	vector<double> samples(100 * chainsize[0] * chainsize[1]);
	vector<int> dims = { chainsize[0],chainsize[1],100 };

	shared_ptr<Session> session = make_shared<Session>(graph);
	map<string, shared_ptr<Tensor>> feedDic;

	feedDic = { { "x", make_shared<Tensor>(dims,samples) } };
	session->run(feedDic, true, 100);
	auto vis = *(*dynamic_pointer_cast<ct::Storage>(graph->storages["visibles_raw"].lock())).storage[100];
	feedDic = { {"x", make_shared<Tensor>(vis)} };
	for (int i = 0; i < lambdas.size(); i++) {
		double l = lambdas[i];
		double k = kappas[0];
		//int counter = 0;
		ofstream output("phi2dmctestNN_lambda=" + to_string(l) + "_" + to_string(chainsize[0]) + ".csv");
		//take first value as base
		double baseValue = 0.1;
		bool first = true;
		int brokenCounter = 0;
		double oldB3 = 0;
		while (true) {
			kappa->value = make_shared<Tensor>(Tensor({ 1 }, { k }));
			lambda->value = make_shared<Tensor>(Tensor({ 1 }, { l }));
			
			double absAvg = 0;
			double vev = 0;
			double absphi4 = 0;
			double absphi2 = 0;
			session->run(feedDic, true, 10);
			auto visibles = *(*dynamic_pointer_cast<ct::Storage>(graph->storages["visibles_raw"].lock())).storage[10];
			auto hiddens = *(*dynamic_pointer_cast<ct::Storage>(graph->storages["hiddens_raw"].lock())).storage[10];
			int counter2 = 0;
			
#pragma omp parallel for reduction(+:absAvg, absphi4, absphi2, counter2, vev)
			for (int b = 0; b < 100; b++) {
				double tmpAvg = 0;
				double tmpphi4 = 0;
				double tmpphi2 = 0;
				double tmpvev = 0;
				int counter = 0;
				
#pragma omp parallel for reduction(+:tmpAvg,tmpphi4,tmpphi2,counter, counter2,tmpvev)
				for (int i = 0; i < visibles.dimensions[0]; i++) {
#pragma omp parallel for reduction(+:tmpAvg,tmpphi4,tmpphi2,counter,counter2,tmpvev)
					for (int j = 0; j < visibles.dimensions[1]; j++)
					{
						if ((i % 2 == 0 && j % 2 == 0) || (i + j) % 2 == 0) {
							auto v1 = visibles[{i, j, b}];
							auto phi4 = (pow(v1, 4));
							auto phi2 = (pow(v1, 2));
							tmpAvg += abs((v1));
							tmpvev += v1;
							tmpphi4 += phi4;
							tmpphi2 += phi2;
							counter ++;
						}
						else {
							//auto v1 = visibles[{i, j, b}];
							int newI = (1.0 / 4.0*(2 * (i - j - 1) + visibles.dimensions[0])); 
							int newJ = (1.0 / 4.0*(2 * (i + j + 1) - visibles.dimensions[1]));
							auto h = hiddens[{newI, newJ, b}];
							auto phi4 = pow(h, 4);
							auto phi2 = pow(h, 2);
							tmpAvg += abs(h);
							tmpvev += h;
							tmpphi2 += phi2;
							tmpphi4 += phi4;
							
							counter++;
							//std::cout << h << std::endl;

						}
						
						//std::cout << tmpAvg << std::endl;
					}
				}
				tmpvev /= counter;
				tmpphi4 /= counter;
				tmpphi2 /= counter;
				tmpAvg /=  counter;
				absAvg += abs(tmpAvg);
				vev += abs(tmpvev);
				absphi4 += tmpphi4;
				absphi2 += tmpphi2;
				counter2++;
			}
			absAvg /= counter2;
			absphi4 /= counter2;
			absphi2 /= counter2;
			vev /= counter2;
			//counter = 0;
			if (first)
			{
				baseValue = absAvg * 10;
				first = false;

			}
			if (k>0.34) {
				brokenCounter++;
				if (brokenCounter > 10) break;
			}
			
			if (oldB3 == 0) {
				oldB3 = absphi4 / (absphi2*absphi2);
			}
			else {
				if (absphi4 / (absphi2*absphi2) - oldB3 < 0.00001) {
					k += 0.01;
					oldB3 = 0;
					output << l << "," << k << "," << absphi4 / (absphi2*absphi2) << "," << absphi2 - (vev*vev) << std::endl;
					std::cout << "<phi^4> " << absphi4 << "  <phi^2> " << absphi2 << "  <phi^2>^2 " << absphi2 * absphi2 << " vev: " << vev << std::endl;
					std::cout << counter2 << "  At l=" << l << " k=" << k << " with B_3=" << absphi4 / (absphi2 * absphi2) << "  susceptibility" << absphi2 - (vev*vev) << std::endl;
				}
				oldB3 = absphi4 / (absphi2*absphi2);
			}
			//k += 0.001;
			feedDic = { {"x" , make_shared<Tensor>(visibles)} };

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
