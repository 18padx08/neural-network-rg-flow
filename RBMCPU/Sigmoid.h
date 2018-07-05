#pragma once
#include "Operation.h"
#include <cmath>
namespace ct {
	class Sigmoid : public Operation
	{
	public:
		Sigmoid(shared_ptr<Node> input);
		~Sigmoid();

		// Inherited via Operation
		virtual shared_ptr<Tensor> compute(std::initializer_list<weak_ptr<Tensor>> input) override;
		virtual shared_ptr<Tensor> compute(std::vector<weak_ptr<Tensor>> input) override;
	};
}

