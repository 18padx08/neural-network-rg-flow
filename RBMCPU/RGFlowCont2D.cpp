#include "RGFlowCont2D.h"
#include <random>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <limits>
#include <chrono>
#include <iostream>






namespace ct {

	double ct::RGFlowCont2D::NormalDist(double mu, double sigma)
	{
		static unsigned seed = std::chrono::system_clock::now().time_since_epoch().count() + rand();
		static std::default_random_engine engine(seed);
		std::normal_distribution<double> n(mu, sigma);
		return n(engine);
	}
	double ct::RGFlowCont2D::UniformDist(double min, double max) {
		static unsigned seed = std::chrono::system_clock::now().time_since_epoch().count() + rand();
		static std::default_random_engine engine(seed);
		std::uniform_real_distribution<double> dist(min, max);
		return dist(engine);
	}

	ct::RGFlowCont2D::RGFlowCont2D(weak_ptr<Node> input, weak_ptr<Variable> kappa, weak_ptr<Variable> Av, weak_ptr<Variable> Ah, weak_ptr<Variable> lambda, bool isInverse) : isInverse(isInverse)
	{
		srand(time(NULL));
		this->inputs.push_back(input);
		this->inputs.push_back(kappa);
		this->inputs.push_back(Av);
		this->inputs.push_back(Ah);
		this->inputs.push_back(lambda);
	}


	RGFlowCont2D::~RGFlowCont2D()
	{
	}

	shared_ptr<Tensor> ct::RGFlowCont2D::compute(std::initializer_list<weak_ptr<Tensor>> input)
	{
		std::vector<weak_ptr<Tensor>> vec(input.begin(), input.end());
		return compute(vec);
	}

	weak_ptr<Variable> ct::RGFlowCont2D::getVarForName(string name, std::vector<weak_ptr<Node>> input) {
		int i = 0;

		for (auto && ele : input) {
			shared_ptr<Variable> tmp = dynamic_pointer_cast<Variable>(ele.lock());
			if (!tmp) continue;
			if (tmp->name == name) {
				return tmp;
			}
			i++;
		}
		return weak_ptr<Variable>();
	}

	double ct::RGFlowCont2D::gauss(double x, double mean, double var, double amplitude) {
		return amplitude * exp(-(1.0 / (2 * var*var))* pow((x - mean), 2));
	}
	double ct::RGFlowCont2D::nongauss(double x, double lambda, double mean, double var) {
		return exp(-(1.0 / (2 * var*var))* pow((x - mean), 2) - lambda * pow((x*x - 1.0), 2));
	}
	shared_ptr<Tensor> ct::RGFlowCont2D::compute(std::vector<weak_ptr<Tensor>> input)
	{
		static double thesquareroot = sqrt(2);
		auto inputTensor = *((this->inputs[0].lock())->output);

		int xDim = inputTensor.dimensions[0];
		int yDim = inputTensor.dimensions[1];
		int samples = inputTensor.dimensions.size() > 2 ? inputTensor.dimensions[2] : 1;
		double kappa = 0;
		double Ah = 0;
		double Av = 0;
		double lambda = 0;
		kappa = (double)*(getVarForName("kappa", this->inputs).lock())->value;
		Ah = (double)*(getVarForName("Ah", this->inputs).lock())->value;
		Av = (double)*(getVarForName("Av", this->inputs).lock())->value;
		if ((getVarForName("lambda", this->inputs).lock()) != nullptr) {
			lambda = *(getVarForName("lambda", this->inputs).lock())->value;
			if (abs(lambda) < 1e-6) {
				lambda = 0;
			}
		}
		//in order to increase acceptance we need to calculate the zeros of the poly of deg 3
		double p = (2.0 - 4 * lambda) / (4 * lambda);
		double p3 = pow(p, 3);

		//2d case means we couple 4 spins to one hidden unit
		if (!isInverse) {
			Tensor tens({ xDim / 2, yDim/2, samples,2 });
#pragma omp parallel for
			for (int s = 0; s < samples; s++) {
#pragma omp parallel for
				for (int i = 0; i < xDim / 2; i++) {
#pragma omp parallel for
					for (int j = 0; j < yDim / 2; j++) {
						//for our case of the crbm we have to draw the new values from a gaussian distribution
						//with mean = 2k/Ah *v_i and sigma of 1/sqrt(Ah)
						int newI = i+j;
						int newJ = j-i + (xDim-2)/2;
						auto val1 = inputTensor[{newI-1,newJ,s}];
						auto val2 = inputTensor[{newI +1,newJ,s}];
						auto val3 = inputTensor[{newI,newJ-1,s}];
						auto val4 = inputTensor[{newI, newJ+1, s}];

						auto mean = (1.0/1.0)*kappa * (1.0 / Av)*(val1 + val2+val3+val4);
						auto variance = sqrt(1.0 / abs(Av)) * 1.0 / thesquareroot;

						auto meanGauss = mean;
						auto varianceGauss = variance;
						auto amplitude = 1.0;
						if (abs(mean) > 0.5 && abs(lambda) > 0) {
							double q = (-2 * mean) / (4 * lambda);
							double D = p3 / 27.0 + pow(q / 2, 2);
							if (D > 0) {
								//we have only one solution
								meanGauss = pow(-q / 2 + sqrt(D), 1.0 / 3) + pow(-q / 2 - sqrt(D), 1.0 / 3);
								amplitude = nongauss(meanGauss, lambda, mean, variance) / gauss(meanGauss, meanGauss, variance);
							}
							else if (D < 0) {
								//we have three real solutions corresponding to the two peaks and the dip
								auto rho = sqrt(-p3 / 27.0);
								auto theta = acos(-q / (2 * rho));
								auto preFactor = 2 * (cbrt(rho));
								auto y1 = preFactor*cos(theta/3);
								auto y2 = preFactor * cos(theta / 3 + 2 * 3.14159 / 3.0);
								auto y3 = preFactor * cos(theta / 3 + 4 * 3.14159 / 3.0);

								//now check for which solution the prob function is maximal
								double argmax = y1;
								double finalValue = nongauss(y1, lambda, mean, variance);
								auto tmp = nongauss(y2, lambda, mean, variance);
								if ( tmp > finalValue) {
									finalValue = tmp;
									argmax = y2;
									tmp = nongauss(y3, lambda, mean, variance);
									if ( tmp> finalValue) {
										argmax = y3;
										finalValue = tmp;
									}
								}
								else {
									tmp = nongauss(y3, lambda, mean, variance);
									if (tmp> finalValue) {
										argmax = y3;
										finalValue = tmp;
									}
								}
								//we need to double the variance to get the small peak 
								///TODO find a better method to determine the new value of gaussVariance
								varianceGauss *= 2;
								meanGauss = argmax;
								amplitude = nongauss(meanGauss, lambda, meanGauss, varianceGauss);
							}
						}

						auto tmp5 = NormalDist(meanGauss, varianceGauss);

						double acceptance = min(1.0, nongauss(tmp5, lambda, mean, variance) / gauss(tmp5, meanGauss, varianceGauss, amplitude));
						if (acceptance < 1) {
							double p = UniformDist(0, 1);
							while (p > acceptance) {
								tmp5 = NormalDist(meanGauss, varianceGauss);
								p = UniformDist(0, 1);
								acceptance = min(1.0, nongauss(tmp5, lambda, mean, variance) / gauss(tmp5, meanGauss, varianceGauss, amplitude));
							}
						}
						tens[{i,j, s, 0}] = tmp5;
					}

				}
			}
			//gaussNumbers.push_back(theGaus);
			return make_shared<Tensor>(tens);
		}
		else {
			// 0 -> 0,2 ; 1-> 2 4
			Tensor tens({ xDim * 2,yDim*2, samples,2 });
#pragma omp parallel for
			for (int s = 0; s < samples; s++) {
#pragma omp parallel for
				for (int i = 0; i < xDim * 2; i++) {
#pragma omp parallel for
					for (int j = 0; j < yDim * 2; j++) {
						if (i % 2 == 0 && j%2 ==0 || (i+j) %2 ==0) {
							//we calculate (i, j+1)
							int newI = (2 * (i - j) + (2 * xDim)) * 1.0 / 4.0;
							int newJ = (2 * (i + 2 + j) - (2 * yDim)) *1.0 / 4.0;
							auto val1 = inputTensor[{newI,newJ, s}];
							auto val2 = inputTensor[{newI-1, newJ, s}];
							auto val3 = inputTensor[{newI, newJ+1, s}];
							auto val4 = inputTensor[{newI-1, newJ+1, s}];

							auto mean = (1.0 / 1.0)*kappa * (1.0 / Av)*(val1 + val2 + val3 + val4);
							auto variance = sqrt(1.0 / abs(Ah)) *1.0 / thesquareroot;
							auto meanGauss = mean;
							auto varianceGauss = variance;
							auto amplitude = 1.0;
							if (abs(mean) > 0.5 && abs(lambda) > 0) {
								double q = (-2 * mean) / (4 * lambda);
								double D = p3 / 27.0 + pow(q / 2, 2);
								if (D > 0) {
									//we have only one solution
									meanGauss = pow(-q / 2 + sqrt(D), 1.0 / 3) + pow(-q / 2 - sqrt(D), 1.0 / 3);
									amplitude = nongauss(meanGauss, lambda, mean, variance) / gauss(meanGauss, meanGauss, variance);
								}
								else if (D < 0) {
									//we have three real solutions corresponding to the two peaks and the dip
									auto rho = sqrt(-p3 / 27.0);
									auto theta = acos(-q / (2 * rho));
									auto preFactor = 2 * (cbrt(rho));
									auto y1 = preFactor * cos(theta / 3);
									auto y2 = preFactor * cos(theta / 3 + 2 * 3.14159 / 3.0);
									auto y3 = preFactor * cos(theta / 3 + 4 * 3.14159 / 3.0);

									//now check for which solution the prob function is maximal
									double argmax = y1;
									double finalValue = nongauss(y1, lambda, mean, variance);
									auto tmp = nongauss(y2, lambda, mean, variance);
									if (tmp > finalValue) {
										finalValue = tmp;
										argmax = y2;
										tmp = nongauss(y3, lambda, mean, variance);
										if (tmp> finalValue) {
											argmax = y3;
											finalValue = tmp;
										}
									}
									else {
										tmp = nongauss(y3, lambda, mean, variance);
										if (tmp> finalValue) {
											argmax = y3;
											finalValue = tmp;
										}
									}
									//we need to double the variance to get the small peak 
									///TODO find a better method to determine the new value of gaussVariance
									varianceGauss *= 2;
									meanGauss = argmax;
									amplitude = nongauss(meanGauss, lambda, meanGauss, varianceGauss);
								}
							}
							auto tmp5 = NormalDist(meanGauss, varianceGauss);
							//theGaus[i] = tmp5;
							double acceptance = min(1.0, nongauss(tmp5, lambda, mean, variance) / gauss(tmp5, meanGauss, varianceGauss, amplitude));
							if (acceptance < 1) {
								double p = UniformDist(0, 1);
								while (p > acceptance) {
									tmp5 = NormalDist(meanGauss, varianceGauss);
									p = UniformDist(0, 1);
									acceptance = min(1.0, nongauss(tmp5, lambda, mean, variance) / gauss(tmp5, meanGauss, varianceGauss, amplitude));
								}
							}

							tens[{i,j, s, 0}] = tmp5;// ((2.0 * 3.14159) / (Av));
												   //tens[{i, s, 1}] = tmp3 * tmp4 / ((2 * 3.14159) / Av);

						}
						else {
							tens[{i,j, s, 0}] = 0;
							tens[{i,j, s, 1}] = 0;
						}
					}
				}
			}
			//gaussNumbers.push_back(theGaus);
			return make_shared<Tensor>(tens);
		}
		return shared_ptr<Tensor>();
	}
	string RGFlowCont2D::type()
	{
		return "rg_flow_cont";
	}
	void RGFlowCont2D::printGaussNumbers(ofstream & log)
	{
		for (auto v : gaussNumbers) {
			for (auto n : v) {
				log << n << ",";
			}
			log << std::endl;
		}
		gaussNumbers.clear();
	}
}
