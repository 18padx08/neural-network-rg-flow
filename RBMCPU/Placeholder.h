#pragma once
#include "Node.h"
#include <string>

namespace ct {
	class Placeholder : public Node
	{
	public:
		Placeholder(std::string name);
		~Placeholder();
		std::string name;
		// Inherited via Node
		virtual shared_ptr<Tensor> compute(std::initializer_list<weak_ptr<Tensor>> input) override;

		// Inherited via Node
		virtual string type() override;

		// Inherited via Node
		virtual shared_ptr<Tensor> compute(std::vector<weak_ptr<Tensor>> input) override;
	};
}
