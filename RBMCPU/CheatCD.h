#pragma once
#include <iostream>
#include <random>
#include <algorithm>
#include <time.h>
#include <cmath>
#include "Graph.h"
#include "Storage.h"
#include "Operation.h"
#include "Variable.h"
#include "OptPlaceholder.h"
namespace ct {
	namespace optimizers {
		class CheatCD : public Operation
		{
		private:
			shared_ptr<Graph> theGraph;
			double learningRate;
			double momentum;
			double lastUpdate;
			uniform_int_distribution<int> dist;
			default_random_engine engine;
		public:
			CheatCD(shared_ptr<Graph> graph);
			~CheatCD();

			// Inherited via Operation
			virtual shared_ptr<Tensor> compute(std::initializer_list<shared_ptr<Tensor>> input) override;
			virtual shared_ptr<Tensor> compute(std::vector<shared_ptr<Tensor>> input) override;
			void optimize();
		};
	}
}

