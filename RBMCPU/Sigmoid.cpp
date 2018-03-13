#include "Sigmoid.h"

namespace ct {

	Sigmoid::Sigmoid(shared_ptr<Node> input)
	{
		this->inputs.push_back(input);
	}


	Sigmoid::~Sigmoid()
	{
	}

	shared_ptr<Tensor> ct::Sigmoid::compute(std::initializer_list<shared_ptr<Tensor>> input)
	{
		vector<shared_ptr<Tensor>> blub(input.begin(), input.end());
		compute(blub);
	}

	shared_ptr<Tensor> ct::Sigmoid::compute(std::vector<shared_ptr<Tensor>> input)
	{
		auto functor = [](double arg) { return 1.0 / (1 + std::exp(-arg)); };
		//blub[0] = make_shared<Tensor>(blub[0]->elementWise(functor));
		return make_shared<Tensor>(input[0]->elementWise(functor));
	}
}
