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
			auto vis_0 = *((dynamic_pointer_cast<Storage>(theGraph->storages["visibles_pooled"]))->storage[0]);
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
			double exp_vis0 = 0;
			double exp_visn = 0;
			double vishid0 = 0;
			double vishidn = 0;
#pragma omp parallel for reduction(+:delta, vishid0, vishidn,exp_vis0,exp_visn, exp_hid0, exp_hidn)
			for (int s = 0; s < samples; s++) {
#pragma omp parallel for reduction(+:delta,vishid0, vishidn,exp_vis0,exp_visn, exp_hid0, exp_hidn)
				for (int i = 0; i < hidDimx; i++) {
					auto corrNNN = vis_0[{2 * i,s}] * hid_0[{i,s}];
					auto corrNNN_n = vis_n[{2 * i,s}] * hid_n[{i,s}];
					//auto corrNNN_n_delta = vis_n[{2 * i, s,1}] * vis_n[{(2 * i + 2) % visDimx, s,1}];
					delta += (corrNNN - corrNNN_n);
					vishidn += corrNNN_n;
					vishid0 += corrNNN;
					if (isCont) {
						exp_vis0 += pow(vis_0[{2*i, s}], 2);
						exp_visn += pow(vis_n[{2*i, s}], 2);
						exp_hid0 += pow(hid_0[{i, s}], 2);
						exp_hidn += pow(hid_n[{i, s}], 2);
					}
				}
			}
			//take the average
			vishidn /= hidDimx * samples;
			vishid0 /= hidDimx * samples;
			exp_hid0 /= hidDimx*samples;
			exp_hidn /= hidDimx*samples;
			exp_vis0 /= hidDimx * samples;
			exp_visn /= hidDimx * samples;
			delta /= hidDimx * samples;
			auto kappa = theGraph->getVarForName("kappa");
			auto Ah = theGraph->getVarForName("Ah");
			auto Av = theGraph->getVarForName("Av");
		
			*kappa->value = *kappa->value + Tensor({1}, {learningRate * 2*(vishid0 - vishidn)});
			*Av->value = *Av->value + Tensor({ 1 }, { -learningRate*(exp_vis0 - exp_visn) });
			*Ah->value = *Ah->value + Tensor({ 1 }, { -learningRate*(exp_hid0 - exp_hidn) });
			kappa.reset();
			Av.reset();
			Ah.reset();
		}
	}
}
