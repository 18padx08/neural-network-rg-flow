#pragma once
#include <random>
#include <algorithm>
#include <time.h>
#include "Operation.h"
#include <iostream>
namespace ct {
	class ProbPooling :public Operation
	{
	public:
		ProbPooling(shared_ptr<Node> input);
		~ProbPooling();
		std::uniform_real_distribution<double> dist;
		std::default_random_engine engine;

		// Inherited via Operation
		virtual shared_ptr<Tensor> compute(std::initializer_list<shared_ptr<Tensor>> input) override;
		virtual shared_ptr<Tensor> compute(std::vector<shared_ptr<Tensor>> input) override;
	};
}
