#pragma once
#include <iostream>
#include <fstream>

#include "Operation.h"
#include "Variable.h"

namespace ct {
	class RGFlowCont : public Operation
	{
	private:
		bool isInverse = false;
		weak_ptr<Variable> getVarForName(string name, std::vector<weak_ptr<Node>> input);
		vector<vector<double>> gaussNumbers;
	public:		
		RGFlowCont(weak_ptr<Node> input, weak_ptr<Variable> kappa, weak_ptr<Variable> Av, weak_ptr<Variable> Ah, weak_ptr<Variable> lambda, bool isInverse);
		~RGFlowCont();

		// Inherited via Operation
		virtual shared_ptr<Tensor> compute(std::initializer_list<weak_ptr<Tensor>> input) override;
		virtual shared_ptr<Tensor> compute(std::vector<weak_ptr<Tensor>> input) override;
		virtual string type() override;
		void printGaussNumbers(ofstream& log);
	};
}

