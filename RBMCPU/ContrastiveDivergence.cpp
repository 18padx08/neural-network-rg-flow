#include "ContrastiveDivergence.h"

namespace ct {
	namespace optimizers {

		ContrastiveDivergence::ContrastiveDivergence(shared_ptr<Graph> graph, double learningRate, double momentum):
			learningRate(learningRate), momentum(momentum)
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
			auto vis_0 = *((dynamic_pointer_cast<Storage>(theGraph->storages["visibles"]))->storage[0]);
			auto hid_0 = *((dynamic_pointer_cast<Storage>(theGraph->storages["hiddens"]))->storage[0]);
			auto vis_n = *((dynamic_pointer_cast<Storage>(theGraph->storages["visibles"]))->storage[k]);
			auto hid_n = *((dynamic_pointer_cast<Storage>(theGraph->storages["hiddens"]))->storage[k]);

			auto visDimx = vis_0.dimensions[0];
			auto hidDimx = hid_0.dimensions[0];

			double delta = 0;
			int counter = 0;
#pragma omp parallel for
			for (int i = 0; i < visDimx; i++) {
				if (i % 2 == 0) {
					if (i == 0) {
						delta += learningRate * (vis_0[{i}] * hid_0[{i / 2}] - vis_n[{i}] * hid_n[{i / 2}]);
						counter++;
					}
					else {
						delta += learningRate * (vis_0[{i}] * hid_0[{i / 2 - 1}] + vis_0[{i}] * hid_0[{i / 2}] - vis_n[{i}] * hid_n[{i / 2 - 1}] - vis_n[{i}] * hid_n[{i / 2}]);
						counter += 2;
					}
				}
			}
			//take the average
			delta /= counter;
			//update the coupling for now only one
			auto coupling = dynamic_pointer_cast<Variable>(theGraph->variables[0]);
			*(coupling->value) = *(coupling->value) + Tensor({ 1 }, { delta });
		}
	}
}
