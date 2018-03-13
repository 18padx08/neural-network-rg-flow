#pragma once
#include "Operation.h"
namespace ct {
	class Sigmoid : public Operation
	{
	public:
		Sigmoid(shared_ptr<Node> input);
		~Sigmoid();

		// Inherited via Operation
		virtual shared_ptr<Tensor> compute(std::initializer_list<shared_ptr<Tensor>> input) override;
		virtual shared_ptr<Tensor> compute(std::vector<shared_ptr<Tensor>> input) override;
	};
}

