#include "Sigmoid.h"

namespace ct {

	Sigmoid::Sigmoid(shared_ptr<Node> input)
	{
		this->inputs.push_back(input);
	}


	Sigmoid::~Sigmoid()
	{
	}

	shared_ptr<Tensor> ct::Sigmoid::compute(std::initializer_list<weak_ptr<Tensor>> input)
	{
		vector<weak_ptr<Tensor>> blub(input.begin(), input.end());
		compute(blub);
	}

	shared_ptr<Tensor> ct::Sigmoid::compute(std::vector<weak_ptr<Tensor>> input)
	{
		auto functor = [](double arg) { return 2.0 / (1.0 + std::exp(-arg)) -1.0; };
		//blub[0] = make_shared<Tensor>(blub[0]->elementWise(functor));
		return make_shared<Tensor>(input[0].lock()->elementWise(functor));
	}
}
