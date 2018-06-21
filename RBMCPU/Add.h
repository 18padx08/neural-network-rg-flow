#pragma once
#include "Operation.h"
#include "Tensor.h"
namespace ct {
	class Add : public Operation
	{
	public:
		Add(shared_ptr<Node> left, shared_ptr<Node> right);
		~Add();

		// Inherited via Operation
		virtual shared_ptr<Tensor> compute(std::initializer_list<weak_ptr<Tensor>> input) override;
		virtual shared_ptr<Tensor> compute(std::vector<weak_ptr<Tensor>> input) override;
	};
}

