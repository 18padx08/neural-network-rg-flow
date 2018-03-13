#pragma once
#include "Node.h"

namespace ct {
	class Storage : public Node
	{
	public:
		Storage();
		~Storage();

		// Inherited via Node
		virtual shared_ptr<Tensor> compute(std::initializer_list<shared_ptr<Tensor>> input) override;
		virtual string type() override;

		// Inherited via Node
		virtual shared_ptr<Tensor> compute(std::vector<shared_ptr<Tensor>> input) override;
	};
}

