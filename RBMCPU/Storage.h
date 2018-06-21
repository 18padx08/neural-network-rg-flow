#pragma once
#include "Node.h"

namespace ct {
	class Storage : public Node
	{
	public:
		Storage(weak_ptr<Node> input, string name);
		~Storage();
		string name;
		// Inherited via Node
		virtual shared_ptr<Tensor> compute(std::initializer_list<weak_ptr<Tensor>> input) override;
		virtual string type() override;

		// Inherited via Node
		virtual shared_ptr<Tensor> compute(std::vector<weak_ptr<Tensor>> input) override;
		vector<shared_ptr<Tensor>> storage;
	};
}

