#include "ModCD.h"


namespace ct {
	namespace optimizers {

		ct::optimizers::ModCD::ModCD(shared_ptr<Graph> graph, double learningRate, double momentum) : learningRate(learningRate), momentum(momentum)
		{
			this->theGraph = graph;
		}

		ct::optimizers::ModCD::~ModCD()
		{
		}

		shared_ptr<Tensor> ct::optimizers::ModCD::compute(std::initializer_list<shared_ptr<Tensor>> input)
		{
			return shared_ptr<Tensor>();
		}

		shared_ptr<Tensor> ct::optimizers::ModCD::compute(std::vector<shared_ptr<Tensor>> input)
		{
			return shared_ptr<Tensor>();
		}

		void ct::optimizers::ModCD::optimize(double learningRate, int k)
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
					auto pos1 = vis_0[{2 * i, s}] * hid_0[{i, s}];
					auto neg1 = vis_n[{2 * i, s}] * hid_n[{i, s}];
					auto pos2 = vis_0[{2 * i + 2, s}] * hid_0[{i, s}];
					auto neg2 = vis_n[{2 * i + 2, s}] * hid_n[{i, s}];
					auto corrNNN = vis_0[{2 * i, s}] * vis_0[{2 * i + 2, s}];
					auto corrNNN_n = vis_n[{2 * i, s}] * vis_n[{2 * i + 2, s}];
					auto corrNNN_nh = hid_n[{i}] * hid_n[{i + 1 < hidDimx ? i + 1 : 0, s}];
					//std::cout << pos1 - neg1 << " " << pos2 - neg2 <<std::endl;
					delta += learningRate * (corrNNN_n - corrNNN);
					test += pos1 * pos2 - neg1 * neg2;
				}
			}
			//take the average
			//std::cout << test / samples/hidDimx << std::endl;
			delta *= learningRate;
			delta /= samples * hidDimx;
			//update the coupling for now only one
			auto coupling = dynamic_pointer_cast<Variable>(theGraph->variables[0]);
			*(coupling->value) = *(coupling->value) + Tensor({ 1 }, { delta }) + Tensor({ 1 }, { lastUpdate * momentum });

			lastUpdate = delta;

		}
	}

}
