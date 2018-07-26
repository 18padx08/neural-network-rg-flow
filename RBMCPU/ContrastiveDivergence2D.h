#pragma once
#include <iostream>
#include <random>
#include <algorithm>
#include <time.h>
#include <math.h>
#include "Graph.h"
#include "Storage.h"
#include "Operation.h"
#include "Variable.h"
#include "OptPlaceholder.h"
namespace ct {
	namespace optimizers {
		class ContrastiveDivergence2D
		{
		private:
			weak_ptr<Graph> theGraph;
			
			double momentum;
			double lastUpdate;
			uniform_int_distribution<int> dist;
			default_random_engine engine;
		public:
			double learningRateL;
			double learningRateK;
			ContrastiveDivergence2D(weak_ptr<Graph> graph, double learningRate = 0.1, double momentum = 0);
			~ContrastiveDivergence2D();

			vector<double> optimize(int k = 1, double betaJ = 1.0, bool useLR = false, bool updateNorms = false, bool fixKappa = false, bool fixLambda = false);

		};
	}
}