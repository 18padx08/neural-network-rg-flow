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
		void ContrastiveDivergence::optimize(int k, double betaJ)
		{
			auto vis_0 = *((dynamic_pointer_cast<Storage>(theGraph->storages["visibles_pooled"]))->storage[0]);
			auto hid_0 = *((dynamic_pointer_cast<Storage>(theGraph->storages["hiddens_raw"]))->storage[0]);
			auto vis_n = *((dynamic_pointer_cast<Storage>(theGraph->storages["visibles_pooled"]))->storage[k]);
			auto hid_n = *((dynamic_pointer_cast<Storage>(theGraph->storages["hiddens_raw"]))->storage[k]);

			auto visDimx = vis_0.dimensions[0];
			auto hidDimx = hid_0.dimensions[0];

			double delta = 0;
			int counter = 0;
			//auto random_connection = dist(engine) % (hidDimx);
			//auto positive = vis_0[{2*random_connection}] * hid_0[{random_connection}];
			//auto negative = vis_n[{2*random_connection}] * hid_n[{random_connection}];
			
			//delta += learningRate * ( positive-negative );	
#pragma omp parallel for reduction(+:delta)
			for (int i = 0; i < hidDimx; i++) {
				auto pos1 = vis_0[{2 * i}] * hid_0[{i}];
				auto neg1 = vis_n[{2 * i}] * hid_n[{i}];
				auto pos2 = vis_0[{2 * i + 2}] * hid_0[{i}];
				auto neg2 = vis_n[{2 * i + 2}] * hid_n[{i}];
				auto corrNNN = vis_0[{2 * i}] * vis_0[{2 * i + 2}];
				auto corrNNN_n = vis_n[{2 * i}] * vis_n[{2 * i + 2}];
				auto corrNNN_nh = hid_n[{i}] * hid_n[{i + 1 < hidDimx? i+1 : 0}];
				//std::cout << pos1 - neg1 << " " << pos2 - neg2 <<std::endl;
				delta += learningRate * (corrNNN_n - betaJ);
			}
			//take the average
			delta /= hidDimx;
			//update the coupling for now only one
			auto coupling = dynamic_pointer_cast<Variable>(theGraph->variables[0]);
			*(coupling->value) = *(coupling->value) + Tensor({ 1 }, {delta }) + Tensor({ 1 }, {lastUpdate * momentum});
			
			lastUpdate = delta;
		}
	}
}
