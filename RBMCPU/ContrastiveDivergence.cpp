#include "ContrastiveDivergence.h"

namespace ct {
	namespace optimizers {

		ContrastiveDivergence::ContrastiveDivergence(weak_ptr<Graph> graph, double learningRate, double momentum):
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
		void ContrastiveDivergence::optimize(int k, double betaJ, bool useLR, bool updateNorms, bool fixKappa, bool fixLambda)
		{
			auto theGraph = this->theGraph.lock();
			auto vis_0 = *((dynamic_pointer_cast<Storage>(theGraph->storages["visibles_pooled"].lock()))->storage[0]);
			auto spec_0 = *((dynamic_pointer_cast<Storage>(theGraph->storages["visibles_pooled"].lock()))->storage[0]);
			auto hid_0 = *((dynamic_pointer_cast<Storage>(theGraph->storages["hiddens_raw"].lock()))->storage[0]);
			auto vis_n = *((dynamic_pointer_cast<Storage>(theGraph->storages["visibles_raw"].lock()))->storage[k]);
			auto hid_n = *((dynamic_pointer_cast<Storage>(theGraph->storages["hiddens_raw"].lock()))->storage[k]);

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
			double testCorr = 0;
			
		
#pragma omp parallel for reduction(+:delta, vishid0, vishidn,exp_vis0,exp_visn, exp_hid0, exp_hidn)
			for (int s = 0; s < samples; s++) {
#pragma omp parallel for reduction(+:delta,vishid0, vishidn,exp_vis0,exp_visn, exp_hid0, exp_hidn)
				for (int i = 0; i < hidDimx; i++) {
					auto v_01 = vis_0[{2 * i, s}];
					
					auto h_01 = hid_0[{i, s}];
					auto h_02 = hid_0[{i - 1, s}];
					auto v_n1 = vis_n[{2 * i, s}];
					auto h_n1 = hid_n[{i, s}];
					auto h_n2 = hid_n[{i - 1, s}];
					auto corrNNN = v_01 * (h_01 + h_02)/2.0;
					auto corrNNN_n = v_n1*(h_n1 + h_n2)/2.0 ;
					//auto corrNNN_n_delta = vis_n[{2 * i, s,1}] * vis_n[{(2 * i + 2) % visDimx, s,1}];
					delta += (corrNNN - corrNNN_n);
					vishidn += v_n1 * vis_n[{2*i+2,s}];
					vishid0 += 0.5 * (v_01 * vis_0[{2 * i + 2, s}] + vis_0[{2 * i + 1, s}]* vis_0[{2 * i + 3, s}]);
					if (isCont) {
						exp_vis0 += pow(pow(v_01,2)-1,2);
						exp_visn += pow(pow(v_n1, 2)-1,2);
						exp_hid0 += pow(h_01, 2);
						exp_hidn += pow(h_n1, 2);
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
			auto lambda = theGraph->getVarForName("lambda");
			delta = vishid0 - vishidn;
			auto tmpDelta = abs(delta) > 0.2 ? (signbit(delta) ? -0.2 : 0.2) : delta;
			auto tmpVisDelta = abs(exp_vis0 - exp_visn) > 0.2 ? (signbit(exp_vis0 - exp_visn) ? -0.2 : 0.2) : exp_vis0 - exp_visn;
			auto tmpHidDelta = abs(exp_hid0 - exp_hidn) > 0.2 ? (signbit(exp_hid0 - exp_hidn) ? -0.2 : 0.2) : exp_hid0 - exp_hidn;
			if(!fixKappa && !isnan(tmpDelta))
				*kappa->value = *kappa->value + Tensor({1}, {learningRate *(tmpDelta)});
			auto newValue = *Av->value + Tensor({ 1 }, { learningRate * (tmpVisDelta), 0.2 });
			auto newValue2 = *Ah->value + Tensor({ 1 }, { learningRate * (tmpHidDelta) });
			if (updateNorms && !(isnan(newValue) || isnan(newValue2))) {
				*Av->value = newValue;
				*Ah->value = newValue2;
			}
			
			if (lambda != nullptr ) {
				auto tmpLamDelta = abs(exp_vis0 - exp_visn)>0.2? (signbit(exp_vis0 - exp_visn)? -0.2:0.2) : exp_vis0 - exp_visn;//abs(exp_vis0 - exp_visn) > 0.2? (signbit(exp_vis0 - exp_visn) ? -0.2 : 0.2) : exp_vis0-exp_visn; //abs((exp_vis0 - 1)*(exp_vis0 - 1) - (exp_visn - 1)*(exp_visn - 1)) > 0.2 ? (signbit((exp_vis0 - 1)*(exp_vis0 - 1) - (exp_visn - 1)*(exp_visn - 1)) ? -0.2 : 0.2) : (exp_vis0 - 1)*(exp_vis0 - 1) - (exp_visn - 1)*(exp_visn - 1);
				//std::cout << tmpLamDelta << std::endl;
				if(!fixLambda && ! isnan(tmpLamDelta))
					*lambda->value = *lambda->value + Tensor({ 1 }, { -learningRate * (tmpLamDelta) });
			}

			lambda.reset();
			kappa.reset();
			Av.reset();
			Ah.reset();
		}
	}
}
