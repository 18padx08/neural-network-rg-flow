#pragma once
#include "Operation.h"
#include "Variable.h"
namespace ct {
	class RGFlowCont : public Operation
	{
	private:
		bool isInverse = false;
		shared_ptr<Variable> getVarForName(string name, std::vector<shared_ptr<Node>> input);
	public:
		RGFlowCont(shared_ptr<Node> input, shared_ptr<Variable> kappa, shared_ptr<Variable> Av, shared_ptr<Variable> Ah, bool isInverse);
		~RGFlowCont();

		// Inherited via Operation
		virtual shared_ptr<Tensor> compute(std::initializer_list<shared_ptr<Tensor>> input) override;
		virtual shared_ptr<Tensor> compute(std::vector<shared_ptr<Tensor>> input) override;
	};
}

