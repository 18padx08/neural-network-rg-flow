#include "CheatCD.h"

namespace ct {
	namespace optimizers {
		CheatCD::CheatCD(shared_ptr<Graph> graph) : engine(time(NULL)), dist()
		{
			this->theGraph = graph;
		}

		CheatCD::~CheatCD()
		{
		}
		shared_ptr<Tensor> CheatCD::compute(std::initializer_list<weak_ptr<Tensor>> input)
		{
			return shared_ptr<Tensor>();
		}
		shared_ptr<Tensor> CheatCD::compute(std::vector<weak_ptr<Tensor>> input)
		{
			return shared_ptr<Tensor>();
		}
		void CheatCD::optimize()
		{
			auto vis_0 = *((dynamic_pointer_cast<Storage>(theGraph->storages["visibles_pooled"].lock()))->storage[0]);
			auto hid_0 = *((dynamic_pointer_cast<Storage>(theGraph->storages["hiddens_pooled"].lock()))->storage[0]);

			auto visDimx = vis_0.dimensions[0];
			auto hidDimx = hid_0.dimensions[0];
			auto samples = vis_0.dimensions.size() > 1 ? vis_0.dimensions[1] : 1;

			double delta = 0;
			int counter = 0;
			//auto positive = vis_0[{2*random_connection}] * hid_0[{random_connection}];
			//auto negative = vis_n[{2*random_connection}] * hid_n[{random_connection}];

			//delta += learningRate * ( positive-negative );	
			double test = 0;
			
#pragma omp parallel for reduction(+:delta)
			for (int s = 0; s < samples; s++) {
				
				int i = dist(engine) % (hidDimx -1);
				auto corrNNN = vis_0[{2 * i, s}] * vis_0[{(2 * i + 2)%hidDimx, s}];
				
				//std::cout << pos1 - neg1 << " " << pos2 - neg2 <<std::endl;
				delta += corrNNN;
			}
			//take the average
			//std::cout << test / samples/hidDimx << std::endl;
			delta /= samples;
			//std::cout << samples << std::endl;
			delta = std::atanh(std::sqrt(std::abs(delta)));
			//if (delta > 4.0) return;
			//update the coupling for now only one
			auto coupling = dynamic_pointer_cast<Variable>(theGraph->variables[0].lock());
			//if (delta - (double)*(coupling->value) > (double)*(coupling->value) * 0.1) return;
			*(coupling->value) = Tensor({ 1 }, { (double)*(coupling->value) /2.0}) + Tensor({ 1 }, { std::abs(delta) });

			lastUpdate = delta;
		}
	}
}


