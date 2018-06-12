#include "RGFlowCont.h"
#include <random>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <limits>
#include <chrono>
#include <iostream>

std::uniform_real_distribution<double> dist(0, 1);
std::normal_distribution<double> normal(0, 1);


double NormalDist(double mu, double sigma)
{
	static unsigned seed = std::chrono::system_clock::now().time_since_epoch().count() + rand();
	static std::default_random_engine engine(seed);
	std::normal_distribution<double> n(mu, sigma);
	return n(engine);
}
double UniformDist(double min, double max) {
	static unsigned seed = std::chrono::system_clock::now().time_since_epoch().count() + rand();
	static std::default_random_engine engine(seed);
	std::uniform_real_distribution<double> dist(min, max);
	return dist(engine);
}

namespace ct {

	ct::RGFlowCont::RGFlowCont(shared_ptr<Node> input, shared_ptr<Variable> kappa, shared_ptr<Variable> Av, shared_ptr<Variable> Ah,  bool isInverse) : isInverse(isInverse)
	{
		srand(time(NULL));
		this->inputs.push_back(input);
		this->inputs.push_back(kappa);
		this->inputs.push_back(Av);
		this->inputs.push_back(Ah);
	}


	RGFlowCont::~RGFlowCont()
	{
	}

	shared_ptr<Tensor> ct::RGFlowCont::compute(std::initializer_list<shared_ptr<Tensor>> input)
	{
		std::vector<shared_ptr<Tensor>> vec(input.begin(), input.end());
		return compute(vec);
	}

	shared_ptr<Variable> ct::RGFlowCont::getVarForName(string name, std::vector<shared_ptr<Node>> input) {
		int i = 0;
		
		for (auto && ele : input) {
			shared_ptr<Variable> tmp = dynamic_pointer_cast<Variable>(ele);
			if (!tmp) continue;
			if (tmp->name == name) {
				return tmp;
			}
			i++;
		}
		return nullptr;
	}

	double gauss(double x, double mean,double var) {
		return exp(- (1.0/(2*var*var))* pow((x - mean), 2));
	}
	double nongauss(double x, double lambda, double mean, double var) {
		return exp(-(1.0 / (2 * var*var))* pow((x - mean), 2) - lambda * pow((x*x -1),2)) ;
	}
	shared_ptr<Tensor> ct::RGFlowCont::compute(std::vector<shared_ptr<Tensor>> input)
	{
		static double thesquareroot = sqrt(2);
		auto inputTensor = *(this->inputs[0]->output);
		
		int xDim = inputTensor.dimensions[0];
		int samples = inputTensor.dimensions.size() > 1 ? inputTensor.dimensions[1] : 1;
		double kappa = 0;
		double Ah = 0;
		double Av = 0;
		double lambda = 0;
		kappa = (double)*getVarForName("kappa", this->inputs)->value;
		Ah = (double)*getVarForName("Ah", this->inputs)->value;
		Av = (double)*getVarForName("Av", this->inputs)->value;
		if ( getVarForName("lambda", this->inputs)!= nullptr) {
			lambda = *getVarForName("lambda", this->inputs)->value;
		}
		
		//first only 1D
		//this means we couple every second spin

		if (!isInverse) {
			Tensor tens({ xDim / 2, samples,2 });
#pragma omp parallel for
			for (int s = 0; s < samples; s++) {
#pragma omp parallel for
				for (int i = 0; i < xDim / 2; i++) {
					//for our case of the crbm we have to draw the new values from a gaussian distribution
					//with mean = 2k/Ah *v_i and sigma of 1/sqrt(Ah)
					auto val1 = inputTensor[{2 * i}];
					auto val2 = inputTensor[{2 * i + 2}];
					//auto tmp1 = NormalDist(2 * kappa / Ah * val1, sqrt(1.0/abs(Ah)));
					auto tmp5 = NormalDist(kappa  * (1.0/Ah)*(val1 + val2), sqrt(1.0 / abs(Ah)) * 1.0/ thesquareroot);
					if (lambda != 0) {
						double p = UniformDist(0,1);
						double acceptance = min(1.0, nongauss(tmp5, lambda, kappa  * (1.0 / Ah)*(val1 + val2), sqrt(1.0 / abs(Ah)) * 1.0 / thesquareroot));
						while (p > acceptance) {
							tmp5 = NormalDist(kappa  * (1.0 / Ah)*(val1 + val2), sqrt(1.0 / abs(Ah)) * 1.0 / thesquareroot);
							p = UniformDist(0, 1);
							acceptance = min(1.0, nongauss(tmp5, lambda, kappa  * (1.0 / Ah)*(val1 + val2), sqrt(1.0 / abs(Ah)) * 1.0 / thesquareroot));
						}

					}
					
					//std::cout << "val=" << val1 << "   " << "x0=" << 2 * kappa / Ah * val1 << " sigma=" << "1" << "  " << tmp1 << std::endl;
					//auto tmp2 = NormalDist(2 * kappa / Ah * val2, sqrt(1.0/abs(Ah)));
					//auto tmp3 = NormalDist(2 *1.05* kappa / Ah * val1, sqrt(1.0/abs(Ah)));
					//auto tmp4 = NormalDist(2 * 1.05* kappa / Ah * val2, sqrt(1.0/abs(Ah)));
					//std::cout << tmp1 << " " << tmp2 << " " << tmp3 << " " << tmp4 << std::endl;

					//std::cout << "gauss prod: " << tmp5/2 << " -> " << (tmp1 + tmp2) / 2.0 << std::endl;
					tens[{i, s, 0}] = tmp5 ; /// ((2.0 * 3.14159) / Ah);
					//tens[{i, s, 1}] = tmp3 * tmp4 / ((3.14159 * 2) / Ah);
					
				}
			}
			return make_shared<Tensor>(tens);
		}
		else {
			// 0 -> 0,2 ; 1-> 2 4
			Tensor tens({ xDim * 2, samples,2 });
#pragma omp parallel for
			for (int s = 0; s < samples; s++) {
#pragma omp parallel for
				for (int i = 0; i < xDim * 2; i++) {
					if (i % 2 == 0) {
						auto val1 = inputTensor[{i / 2 - 1}];
						auto val2 = inputTensor[{i / 2}];
						//auto tmp1 = NormalDist(2* (kappa) / Av * val1, sqrt(1.0/abs(Av)));
						//auto tmp2 = NormalDist(2 * (kappa) / Av * val2, sqrt(1.0/abs(Av)));
						//auto tmp3 = NormalDist(2 * 1.05* kappa / Av * val1, sqrt(1.0/abs(Av)));
						//auto tmp4 = NormalDist(2 * 1.05* kappa / Av * val2, sqrt(1.0/abs(Av)));
						auto tmp5 = NormalDist(kappa  * (1.0/Av) * (val1 + val2), sqrt(1.0 / abs(Av)) *1.0 / thesquareroot);

						if (lambda != 0) {
							double p = UniformDist(0, 1);
							double acceptance = min(1.0, nongauss(tmp5, lambda, kappa  * (1.0 / Ah)*(val1 + val2), sqrt(1.0 / abs(Ah)) * 1.0 / thesquareroot));
							while (p > acceptance) {
								tmp5 = NormalDist(kappa  * (1.0 / Ah)*(val1 + val2), sqrt(1.0 / abs(Ah)) * 1.0 / thesquareroot);
								p = UniformDist(0, 1);
								acceptance = min(1.0, nongauss(tmp5, lambda, kappa  * (1.0 / Ah)*(val1 + val2), sqrt(1.0 / abs(Ah)) * 1.0 / thesquareroot));
							}

						}

						tens[{i, s, 0}] = tmp5 ;// ((2.0 * 3.14159) / (Av));
						//tens[{i, s, 1}] = tmp3 * tmp4 / ((2 * 3.14159) / Av);
						
					}
					else {
						tens[{i, s, 0}] = 0;
						tens[{i, s, 1}] = 0;
					}
				}
			}
			return make_shared<Tensor>(tens);
		}
		return shared_ptr<Tensor>();
	}
}
