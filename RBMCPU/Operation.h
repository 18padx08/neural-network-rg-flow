#pragma once
#include "Node.h"
namespace ct {
	class Operation : public Node
	{
	public:
		Operation();
		virtual ~Operation();

		// Inherited via Node
		virtual shared_ptr<Tensor> compute(std::initializer_list<shared_ptr<Tensor>> input) = 0;

		// Inherited via Node
		virtual string type() override;

		// Inherited via Node
		virtual shared_ptr<Tensor> compute(std::vector<shared_ptr<Tensor>> input) =0;
	};
}

