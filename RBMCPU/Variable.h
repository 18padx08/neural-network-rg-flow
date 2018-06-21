#pragma once
#include "Node.h"

namespace ct {
	class Variable : public Node
	{
	public:
		Variable();
		~Variable();

		// Inherited via Node
		virtual shared_ptr<Tensor> compute(std::initializer_list<weak_ptr<Tensor>> input) override;

		// Inherited via Node
		virtual string type() override;

		shared_ptr<Tensor> value;
		string name;

		// Inherited via Node
		virtual shared_ptr<Tensor> compute(std::vector<weak_ptr<Tensor>> input) override;
	};
}

