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
		void ContrastiveDivergence::optimize(int k)
		{
			auto vis_0 = *((dynamic_pointer_cast<Storage>(theGraph->storages["visibles_raw"]))->storage[0]);
			auto hid_0 = *((dynamic_pointer_cast<Storage>(theGraph->storages["hiddens_raw"]))->storage[0]);
			auto vis_n = *((dynamic_pointer_cast<Storage>(theGraph->storages["visibles_raw"]))->storage[k]);
			auto hid_n = *((dynamic_pointer_cast<Storage>(theGraph->storages["hiddens_raw"]))->storage[k]);

			auto visDimx = vis_0.dimensions[0];
			auto hidDimx = hid_0.dimensions[0];

			double delta = 0;
			int counter = 0;
			auto random_connection = dist(engine) % (visDimx);
			delta += learningRate * (vis_0[{random_connection}] * hid_0[{random_connection / 2}] - vis_n[{random_connection}] * hid_n[{random_connection / 2}]);	
/*
			for (int i = 0; i < visDimx; i++) {
				if (i % 2 == 0) {
					if (i == 0) {
						delta += learningRate * (vis_0[{i}] * hid_0[{i / 2}] - vis_n[{i}] * hid_n[{i / 2}]);
					}
					else {
						delta += learningRate * (vis_0[{i}] * hid_0[{i / 2 - 1}]  - vis_n[{i}] * hid_n[{i / 2 - 1}] + vis_0[{i}] * hid_0[{i / 2 }] - vis_n[{i}] * hid_n[{i / 2 }]);
						
					}
				}
			}*/
			//take the average
			//delta /= visDimx-1;
			//update the coupling for now only one
			auto coupling = dynamic_pointer_cast<Variable>(theGraph->variables[0]);
			*(coupling->value) = *(coupling->value) + Tensor({ 1 }, { delta }) + Tensor({ 1 }, {lastUpdate * momentum});
			lastUpdate = delta;
		}
	}
}
