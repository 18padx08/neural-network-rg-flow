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
		return make_shared<Tensor>(input[0]->elementWise(functor));
	}
}
