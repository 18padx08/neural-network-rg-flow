#include "RGLayer.h"

namespace ct {

	RGLayer::RGLayer(shared_ptr<Node> input, shared_ptr<Variable> variable, bool isInverse) : isInverse(isInverse)
	{
		this->inputs.push_back(input);
		this->inputs.push_back(variable);
	}


	RGLayer::~RGLayer()
	{
	}

	shared_ptr<Tensor> ct::RGLayer::compute(std::initializer_list<shared_ptr<Tensor>> input)
	{
		std::vector<shared_ptr<Tensor>> vec(input.begin(), input.end());
		return compute(vec);
	}


	shared_ptr<Tensor> ct::RGLayer::compute(std::vector<shared_ptr<Tensor>> input)
	{
		auto inputTensor = *(this->inputs[0]->output);
		int xDim = inputTensor.dimensions[0];
		int samples = inputTensor.dimensions.size() > 1 ? inputTensor.dimensions[1] : 1;
		auto coupling = (double)*(dynamic_pointer_cast<Variable>(this->inputs[1])->value);
		//first only 1D
		//this means we couple every second spin
		
		if (!isInverse) {
			Tensor tens({ xDim / 2, samples,2 });
#pragma omp parallel for
			for (int s = 0; s < samples; s++) {
#pragma omp parallel for
				for (int i = 0; i < xDim / 2; i++) {
					tens[{i,s,0}] = 2.0 / (1 + std::exp(-coupling * (inputTensor[{2 * i,s}] + (2 * i + 2 < xDim ? inputTensor[{2 * i + 2,s}] : 0)))) - 1;
					tens[{i, s, 1}] = 2.0 / (1 + std::exp(-(1.05*coupling) * (inputTensor[{2 * i, s}] + (2 * i + 2 < xDim ? inputTensor[{2 * i + 2, s}] : 0)))) - 1;
				}
			}
			return make_shared<Tensor>(tens);
		}
		else {
			// 0 -> 0,2 ; 1-> 2 4
			Tensor tens({ xDim *2, samples,2});
#pragma omp parallel for
			for (int s = 0; s < samples; s++) {
#pragma omp parallel for
				for (int i = 0; i < xDim * 2; i++) {
					if (i % 2 == 0) {
						if (i == 0) {
							tens[{i,s,0}] = 2.0 / (1 + std::exp(-coupling * inputTensor[{i / 2,s}])) - 1;
							tens[{i,s,1}] = 2.0 / (1 + std::exp(-1.05*coupling * inputTensor[{i / 2, s}])) - 1;
						}
						else {
							tens[{i,s,0}] = 2.0 / (1 + std::exp(-coupling * (inputTensor[{i / 2 - 1,s}] + inputTensor[{i / 2,s }]))) - 1;
							tens[{i, s,1}] = 2.0 / (1 + std::exp(-1.05*coupling * (inputTensor[{i / 2 - 1, s}] + inputTensor[{i / 2, s }]))) - 1;
						}
					}
					else {
						tens[{i,s,0}] = 0;
						tens[{i, s, 1}] = 0;
					}
				}
			}
			return make_shared<Tensor>(tens);
		}
		return shared_ptr<Tensor>();
	}
}
