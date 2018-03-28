#include "ContrastiveDivergence.h"

namespace ct {
	namespace optimizers {

		ContrastiveDivergence::ContrastiveDivergence(shared_ptr<Graph> graph, double learningRate, double momentum):
			learningRate(learningRate), momentum(momentum), engine(time(NULL)), dist()
		{
			theGraph = graph;
		}


		ContrastiveDivergence::~ContrastiveDivergence()
		{
		}

		shared_ptr<Tensor> ct::optimizers::ContrastiveDivergence::compute(std::initializer_list<shared_ptr<Tensor>> input)
		{
			return shared_ptr<Tensor>();
		}

		shared_ptr<Tensor> ct::optimizers::ContrastiveDivergence::compute(std::vector<shared_ptr<Tensor>> input)
		{
			return shared_ptr<Tensor>();
		}
		void ContrastiveDivergence::optimize(int k, double betaJ, bool useLR)
		{
			auto vis_0 = *((dynamic_pointer_cast<Storage>(theGraph->storages["visibles_pooled"]))->storage[0]);
			auto hid_0 = *((dynamic_pointer_cast<Storage>(theGraph->storages["hiddens_pooled"]))->storage[0]);
			auto vis_n = *((dynamic_pointer_cast<Storage>(theGraph->storages["visibles_pooled"]))->storage[k]);
			auto hid_n = *((dynamic_pointer_cast<Storage>(theGraph->storages["hiddens_pooled"]))->storage[k]);

			auto visDimx = vis_0.dimensions[0];
			auto hidDimx = hid_0.dimensions[0];
			auto samples = vis_0.dimensions.size() > 1 ? vis_0.dimensions[1] : 1;

			double delta = 0;
			int counter = 0;
			//auto positive = vis_0[{2*random_connection}] * hid_0[{random_connection}];
			//auto negative = vis_n[{2*random_connection}] * hid_n[{random_connection}];
			
			//delta += learningRate * ( positive-negative );	
			double test = 0;
#pragma omp parallel for reduction(+:delta,test)
			for (int s = 0; s < samples; s++) {
#pragma omp parallel for reduction(+:delta,test)
				for (int i = 0; i < hidDimx; i++) {
					auto corrNNN = vis_0[{2 * i,s}] * vis_0[{(2 * i + 2) % hidDimx,s}];
					auto corrNNN_n = vis_n[{2 * i,s}] * vis_n[{(2 * i + 2) % hidDimx,s}];
					delta += (corrNNN_n - corrNNN);
				}
			}
			//take the average
			//std::cout << delta << std::endl;
			auto coupling = dynamic_pointer_cast<Variable>(theGraph->variables[0]);
			delta /= samples*hidDimx;
			auto blub = pow(cosh(abs(delta)), 2);
			//std::cout << std::endl << blub << std::endl;
			delta *= useLR? learningRate : blub;
			//absolute boundary of delta: not bigger than 10 % of the current value of the coupling (prevent going to far away from 0)
			if (abs(delta) > 0.1 * abs(*(coupling->value))) {
				delta = 0.1 * abs(*(coupling->value)) * (signbit(delta)? -1.0 : 1.0);
			}
			//update the coupling for now only one
			
			*(coupling->value) = *(coupling->value)  + Tensor({ 1 }, {delta}) + Tensor({ 1 }, {lastUpdate * momentum});
			
			lastUpdate = delta;
		}
	}
}
