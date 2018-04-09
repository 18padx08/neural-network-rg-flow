#include "ProbPooling.h"

namespace ct {
	ProbPooling::ProbPooling(shared_ptr<Node> input) : engine(time(NULL)), dist(0.0, 1.0)
	{
		this->inputs.push_back(input);
	}


	ProbPooling::~ProbPooling()
	{

	}

	shared_ptr<Tensor> ct::ProbPooling::compute(std::initializer_list<shared_ptr<Tensor>> input)
	{
		std::vector<shared_ptr<Tensor>> inputs(input.begin(), input.end());
		return compute(inputs);
	}

	shared_ptr<Tensor> ct::ProbPooling::compute(std::vector<shared_ptr<Tensor>> input)
	{

		auto functor = [this](double arg)
		{
			double p = this->dist(this->engine);
			double prob = (arg + 1) / 2.0;
			//std::cout << p << " < " << prob << "?" << std::endl;
			if (p < prob) {
				return 1;
			}
			return -1;
		};
		//blub[0] = make_shared<Tensor>(blub[0]->elementWise(functor));
		Tensor t(*input[0]);
		auto samples = t.dimensions.size() > 1 ? t.dimensions[1] : 1;
#pragma omp parallel for 
		for (int s = 0; s < samples; s++) {
#pragma omp parallel for
			for (int i = 0; i < t.dimensions[0]; i++)
			{
				double *t1 = t.getPointer({i, s, 0});
				double *t2 = t.getPointer({i, s, 1});
				double p = this->dist(this->engine);
				double prob = (*t1+ 1) / 2.0;
				double prob2 = (*t2 + 1) / 2.0;
				//std::cout << p << " < " << prob << "?" << std::endl;
				if (p < prob) {
					*t1 = 1;
				}
				else {
					*t1 = -1;
				}
				if (p < prob2) {
					*t2 = 1;
				}
				else {
					*t2 = -1;
				}
			
			}
		}
		input.clear();
		return make_shared<Tensor>(t);
		//return shared_ptr<Tensor>();
	}
}
