#pragma once
#include <iostream>
#include <random>
#include <algorithm>
#include <time.h>
#include "Graph.h"
#include "Storage.h"
#include "Operation.h"
#include "Variable.h"
namespace ct {
	namespace optimizers {
		class ContrastiveDivergence : public Operation
		{
		private:
			shared_ptr<Graph> theGraph;
			double learningRate;
			double momentum;
			double lastUpdate;
			uniform_int_distribution<int> dist;
			default_random_engine engine;
		public:
			ContrastiveDivergence(shared_ptr<Graph> graph, double learningRate = 0.1, double momentum = 0);
			~ContrastiveDivergence();

			// Inherited via Operation
			virtual shared_ptr<Tensor> compute(std::initializer_list<shared_ptr<Tensor>> input) override;
			virtual shared_ptr<Tensor> compute(std::vector<shared_ptr<Tensor>> input) override;
			void optimize(int k=1);
		};
	}
}
