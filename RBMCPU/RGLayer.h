#pragma once
#include "Operation.h"
#include "Variable.h"
#include <cmath>
namespace ct {
	class RGLayer : public Operation
	{
	private: 
		bool isInverse = false;
	public:
		RGLayer(shared_ptr<Node> input, shared_ptr<Variable> variable, bool isInverse);
		~RGLayer();

		// Inherited via Operation
		virtual shared_ptr<Tensor> compute(std::initializer_list<weak_ptr<Tensor>> input) override;
		virtual shared_ptr<Tensor> compute(std::vector<weak_ptr<Tensor>> input) override;
	};
}

