#pragma once
#include <iostream>
#include <fstream>

#include "Operation.h"
#include "Variable.h"
using namespace ct;
namespace ct {
	class RGFlowCont2D : public Variable
	{
	private:
		bool isInverse = false;
		weak_ptr<Variable> getVarForName(string name, std::vector<weak_ptr<Node>> input);
		vector<vector<double>> gaussNumbers;

		double gauss(double x, double mean, double var, double amplitude = 1.0);
		double nongauss(double x, double lambda, double mean, double var);
		double NormalDist(double mu, double sigma);
		double UniformDist(double min, double max);
	public:
		RGFlowCont2D(weak_ptr<Node> input, weak_ptr<Variable> kappa, weak_ptr<Variable> Av, weak_ptr<Variable> Ah, weak_ptr<Variable> lambda, bool isInverse);
		~RGFlowCont2D();

		// Inherited via Operation
		virtual shared_ptr<Tensor> compute(std::initializer_list<weak_ptr<Tensor>> input) override;
		virtual shared_ptr<Tensor> compute(std::vector<weak_ptr<Tensor>> input) override;
		virtual string type() override;
		void printGaussNumbers(ofstream& log);
	};
}
