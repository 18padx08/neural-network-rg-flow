#pragma once
#include <vector>
#include "Node.h"
#include "Tensor.h"
namespace ct {
	//Only use this to close a graph
	class OptPlaceholder : public Node
	{
	public:
		OptPlaceholder(std::string name);
		~OptPlaceholder();
		std::string name;

		// Inherited via Node
		virtual shared_ptr<Tensor> compute(std::initializer_list<weak_ptr<Tensor>> input) override;
		virtual shared_ptr<Tensor> compute(std::vector<weak_ptr<Tensor>> input) override;
		virtual string type() override;
	};
}

