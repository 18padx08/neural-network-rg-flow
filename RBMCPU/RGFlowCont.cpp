#include "RGFlowCont.h"
#include <random>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <limits>
#include <iostream>

std::uniform_real_distribution<double> dist(0, 1);
std::normal_distribution<double> normal(0, 1);
std::default_random_engine engine(time(NULL));

double generateGaussianNoise(double mu, double sigma)
{
	return normal(engine);
}

namespace ct {

	ct::RGFlowCont::RGFlowCont(shared_ptr<Node> input, shared_ptr<Variable> variable, shared_ptr<Variable> scalingParam,  bool isInverse) : isInverse(isInverse)
	{
		this->inputs.push_back(input);
		this->inputs.push_back(variable);
		this->inputs.push_back(scalingParam);
	}


	RGFlowCont::~RGFlowCont()
	{
	}

	shared_ptr<Tensor> ct::RGFlowCont::compute(std::initializer_list<shared_ptr<Tensor>> input)
	{
		std::vector<shared_ptr<Tensor>> vec(input.begin(), input.end());
		return compute(vec);
	}


	shared_ptr<Tensor> ct::RGFlowCont::compute(std::vector<shared_ptr<Tensor>> input)
	{
		auto inputTensor = *(this->inputs[0]->output);
		int xDim = inputTensor.dimensions[0];
		int samples = inputTensor.dimensions.size() > 1 ? inputTensor.dimensions[1] : 1;
		auto v1 = (dynamic_pointer_cast<Variable>(this->inputs[1]));
		auto v2 = (dynamic_pointer_cast<Variable>(this->inputs[2]));
		double coupling = 0;
		double scalingParam = 0;
		if (v1->name == "A") {
			coupling = (double)*(v1->value);
			scalingParam = (double)*(v2->value);
		}
		else {
			coupling = (double)*(v2->value);
			scalingParam = (double)*(v1->value);
		}
		//first only 1D
		//this means we couple every second spin

		if (!isInverse) {
			Tensor tens({ xDim / 2, samples,2 });
#pragma omp parallel for
			for (int s = 0; s < samples; s++) {
#pragma omp parallel for
				for (int i = 0; i < xDim / 2; i++) {
					//A continous RBM needs gaussian noise and a scaling parameter
					auto gaussNoise = 0.5 * generateGaussianNoise(0,2);
					//std::cout << gaussNoise << std::endl;
					tens[{i, s, 0}] = 2.0 / (1 + std::exp(- scalingParam *  (coupling *(inputTensor[{2 * i, s}] + (2 * i + 2 < xDim ? inputTensor[{2 * i + 2, s}] : 0)) + gaussNoise))) - 1;
					tens[{i, s, 1}] = 2.0 / (1 + std::exp(-(1.05*scalingParam) * (coupling * (inputTensor[{2 * i, s}] + (2 * i + 2 < xDim ? inputTensor[{2 * i + 2, s}] : 0)) + gaussNoise))) - 1;
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
						auto gaussNoise =0.5* generateGaussianNoise(0, 2);

						if (i == 0) {
							tens[{i, s, 0}] = 2.0 / (1 + std::exp(-scalingParam *  (coupling *inputTensor[{i / 2, s}] + gaussNoise))) - 1;
							tens[{i, s, 1}] = 2.0 / (1 + std::exp(-1.05* scalingParam* (coupling *inputTensor[{i / 2, s}] + gaussNoise))) - 1;
						}
						else {
							tens[{i, s, 0}] = 2.0 / (1 + std::exp(-scalingParam * ( coupling * (inputTensor[{i / 2 - 1, s}] + inputTensor[{i / 2, s }] )+ gaussNoise))) - 1;
							tens[{i, s, 1}] = 2.0 / (1 + std::exp(-1.05 * scalingParam * (coupling*(inputTensor[{i / 2 - 1, s}] + inputTensor[{i / 2, s }])+ gaussNoise))) - 1;
						}
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
