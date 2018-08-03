#include "Phi2DMCTests.h"



Phi2DMCTests::Phi2DMCTests()
{
}


Phi2DMCTests::~Phi2DMCTests()
{
}

void Phi2DMCTests::criticalLineTest(vector<int> chainsize, vector<double> kappas, vector<double> lambdas, double stepsize, double finalBeta)
{
	
	for (int i = 0; i < lambdas.size(); i++) {
		double l = lambdas[i];
		//double k = kappas[0];
		int counter = 0;
		
		
		for(double k : kappas) {
			ofstream output("phi2dmctest_lambda=" + to_string(l) + "_kappa=" + to_string(k) + "_cs=" + to_string(chainsize[0]) + ".csv");
			Phi2D phi(chainsize, k, l);
			phi.useWolff = true;
			phi.thermalize(400);
			
			double absAvg = 0;
			double phi4 = 0;
			double phi2 = 0;
			double vev = 0;
			for (int i = 0; i < 1000; i++) {
				auto quartPhi = phi.quarticVolumeAverage();
				auto squPhi = phi.squaredVolumeAverage();
				auto absAvg = phi.volumeAverage();
				auto action = phi.getKappaWeight();
				phi4 += phi.quarticVolumeAverage();
				phi2 += phi.squaredVolumeAverage();
				//absAvg += phi.absoluteVolumeAverage();
				vev += abs(phi.volumeAverage());

				output << absAvg << "," << squPhi << "," << quartPhi << "," << action  << std::endl;
				phi.monteCarloSweep();
				phi.monteCarloSweep();
				phi.monteCarloSweep();
				phi.monteCarloSweep();
			}
			vev /= 100;
			absAvg /= 100;
			phi4 /= 100;
			phi2 /= 100;
			
			
			std::cout << "At l=" << l << " k=" << k << " with B_3=" << phi4 / (phi2*phi2) << "   " << phi2 - (vev*vev) << "  vev: " << vev << std::endl;
			
			output.close();
		}
	}
}



void Phi2DMCTests::criticalLineTestNN(vector<int> chainsize, vector<double> kappas, vector<double> lambdas,double stepsize)
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
			if (k>0.8) {
				break;
			}
			
			if (oldB3 == 0) {
				oldB3 = absphi4 / (absphi2*absphi2);
			}
			else {
				if (absphi4 / (absphi2*absphi2) - oldB3 < 0.00001) {
					k += stepsize;
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

void Phi2DMCTests::criticalSlowingDown(vector<int> chainsize, vector<double> kappas, vector<double> lambda, int batchsize)
{
	auto graph = RBMCompTree::getRBM2DGraph();
	auto kap = graph->getVarForName("kappa");
	auto lamb = graph->getVarForName("lambda");

	auto session = make_shared<Session>(graph);
	vector<int> dimensions = { chainsize[0], chainsize[1], batchsize };
	vector<double> samples(chainsize[0] * chainsize[1] * batchsize);

	for (auto k : kappas) {
		for (auto l : lambda) {
			kap->value = make_shared<Tensor>(Tensor({ 1 }, { k }));
			lamb->value = make_shared<Tensor>(Tensor({ 1 }, { l }));
			ofstream output("csd_kappa=" + to_string(k) + "_lambda=" + to_string(l) + "_cs=" + to_string(chainsize[0]) +"_bs=" + to_string(batchsize)+".csv");
			for (int i = 0; i < samples.size(); i++) {
				samples[i] = i % 2 == 0 ? -1 : 1;
			}
			map<string, shared_ptr<Tensor>> feedDic = { {"x" , make_shared<Tensor>(dimensions, samples)} };
			//set kappa and lambda
			for (int j = 0; j < 150; j++) {
				
				session->run(feedDic, true, 1);
				//make measurement
				auto visibles = *(*dynamic_pointer_cast<ct::Storage>(graph->storages["visibles_raw"].lock())).storage[1];
				auto hiddens = *(*dynamic_pointer_cast<ct::Storage>(graph->storages["hiddens_raw"].lock())).storage[1];
				feedDic = { {"x", make_shared<Tensor>(visibles)} };

				double absAvg = 0;
				double vev = 0;
				double absphi4 = 0;
				double absphi2 = 0;
				int counter2 = 0;
#pragma omp parallel for reduction(+:absAvg, absphi4, absphi2, counter2, vev)
				for (int b = 0; b < batchsize; b++) {
					double tmpAvg = 0;
					double tmpphi4 = 0;
					double tmpphi2 = 0;
					double tmpvev = 0;
					int counter = 0;

#pragma omp parallel for reduction(+:tmpAvg,tmpphi4,tmpphi2,counter,tmpvev)
					for (int i = 0; i < visibles.dimensions[0]; i++) {
#pragma omp parallel for reduction(+:tmpAvg,tmpphi4,tmpphi2,counter,tmpvev)
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
								counter++;
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
					tmpAvg /= counter;
					absAvg += abs(tmpAvg);
					vev += abs(tmpvev);
					absphi4 += tmpphi4;
					absphi2 += tmpphi2;
					counter2++;
					//output << tmpvev << "," << tmpphi2 << "," << tmpphi4 << std::endl;
				}
				absAvg /= counter2;
				absphi4 /= counter2;
				absphi2 /= counter2;
				vev /= counter2;
				output << vev << "," << absphi2 << ","<< absphi4 << "," << (absphi2 - pow(vev,2))/sqrt(batchsize) << std::endl;
			}
		}

	}

}

void Phi2DMCTests::operator()(string name, map<string, double> num_vars, map<string, string> str_vars, map<string, vector<double>> list_vars)
{
	vector<int> chainsize = this->getIntVector("chainsize", num_vars, list_vars);
	vector<double> kappas = this->getDoubleVector("kappa", num_vars, list_vars);
	vector<double> lambdas = this->getDoubleVector("lambda", num_vars, list_vars);
	double finalK = num_vars.find("finalKappa") == num_vars.end() ? kappas[0] * 20 : num_vars["finalKappa"];
	double ss = num_vars.find("stepsize") == num_vars.end() ? 0.01 : num_vars["stepsize"];
	int bs = num_vars.find("batchsize") == num_vars.end() ? 0.01 : num_vars["batchsize"];
	if (name == "criticalLineTest") {
		criticalLineTest(chainsize, kappas, lambdas,ss, finalK);
	}
	if (name == "criticalLineNNTest") {
		criticalLineTestNN(chainsize, kappas, lambdas,ss);
	}
	if (name == "criticalSlowingDown") {
		criticalSlowingDown(chainsize, kappas, lambdas, bs);
	}
}
