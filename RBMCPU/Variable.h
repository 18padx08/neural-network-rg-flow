#pragma once
#include "Node.h"

namespace ct {
	class Variable : public Node
	{
	public:
		Variable();
		~Variable();

		// Inherited via Node
		virtual shared_ptr<Tensor> compute(std::initializer_list<shared_ptr<Tensor>> input) override;

		// Inherited via Node
		virtual string type() override;

		shared_ptr<Tensor> value;

		// Inherited via Node
		virtual shared_ptr<Tensor> compute(std::vector<shared_ptr<Tensor>> input) override;
	};
}

