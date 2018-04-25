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
			
			theGraph.reset();
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
			auto vis_0 = *((dynamic_pointer_cast<Storage>(theGraph->storages["visibles_raw"]))->storage[0]);
			auto hid_0 = *((dynamic_pointer_cast<Storage>(theGraph->storages["hiddens_raw"]))->storage[0]);
			auto vis_n = *((dynamic_pointer_cast<Storage>(theGraph->storages["visibles_raw"]))->storage[k]);
			auto hid_n = *((dynamic_pointer_cast<Storage>(theGraph->storages["hiddens_raw"]))->storage[k]);

			auto visDimx = vis_0.dimensions[0];
			auto hidDimx = hid_0.dimensions[0];
			auto samples = vis_0.dimensions.size() > 1 ? vis_0.dimensions[1] : 1;

			double delta = 0;
			int counter = 0;
			//auto positive = vis_0[{2*random_connection}] * hid_0[{random_connection}];
			//auto negative = vis_n[{2*random_connection}] * hid_n[{random_connection}];
			
			//delta += learningRate * ( positive-negative );	
			double test = 0;
			bool isCont = theGraph->variables.size() > 2 ? true : false;
			double exp_hid0 = 0;
			double exp_hidn = 0;
#pragma omp parallel for reduction(+:delta,test, exp_hid0, exp_hidn)
			for (int s = 0; s < samples; s++) {
#pragma omp parallel for reduction(+:delta,test, exp_hid0, exp_hidn)
				for (int i = 0; i < hidDimx; i++) {
					auto corrNNN = vis_0[{2 * i,s}] * vis_0[{(2 * i + 2) % hidDimx,s}];
					auto corrNNN_n = vis_n[{2 * i,s}] * vis_n[{(2 * i + 2) % hidDimx,s}];
					auto corrNNN_n_delta = vis_n[{2 * i, s,1}] * vis_n[{(2 * i + 2) % hidDimx, s,1}];
					delta += (corrNNN_n - corrNNN);
					test += (corrNNN_n_delta - corrNNN);
					if (isCont) {
						exp_hid0 += pow(hid_0[{i, s}], 2);
						exp_hidn += pow(hid_n[{i, s}], 2);
					}
				}
			}
			//take the average
			//std::cout << delta << std::endl;
			auto coupling = dynamic_pointer_cast<Variable>(theGraph->variables[0]);
			if (coupling->name != "A") {
				coupling = dynamic_pointer_cast<Variable>(theGraph->variables[1]);
			}
			delta /= samples*hidDimx;
			test /= samples * hidDimx;
			exp_hid0 /= samples * hidDimx;
			exp_hidn /= samples * hidDimx;
			auto blub =  abs(((test - delta)/0.05));
			//std::cout << test << "  " << delta << std::endl;
			//std::cout << std::endl << blub << std::endl;
			delta *= useLR? learningRate : blub;
			//absolute boundary of delta: not bigger than 20 % of the current value of the coupling (prevent going to far away from 0)
			if (abs(delta) >  abs(*(coupling->value))) {
				delta = 0.2 * abs(*(coupling->value)) * (signbit(delta)? -1.0 : 1.0);
				//std::cout << "some problems indeed";
			}
			//update the coupling for now only one
			if (isCont) {
				auto scaling = dynamic_pointer_cast<Variable>(theGraph->variables[1]);
				if (scaling->name != "s") {
					scaling = dynamic_pointer_cast<Variable>(theGraph->variables[0]);
				}
			//	*(scaling->value) = *(scaling->value) + Tensor({ 1 }, { (1.0 / pow(*(coupling->value),2))*learningRate * (exp_hidn - exp_hid0) });
			}
			//std::cout << std::endl <<  delta << std::endl;
			*(coupling->value) = *(coupling->value)  +  Tensor({ 1 }, {+delta}) + Tensor({ 1 }, {lastUpdate * momentum});
			
			lastUpdate = delta;
			coupling.reset();
		}
	}
}
