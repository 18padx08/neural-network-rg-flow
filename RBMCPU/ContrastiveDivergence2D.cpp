#include "ContrastiveDivergence2D.h"

namespace ct {
	namespace optimizers {

		ContrastiveDivergence2D::ContrastiveDivergence2D(weak_ptr<Graph> graph, double learningRate, double momentum) :
			learningRate(learningRate), momentum(momentum), engine(time(NULL)), dist(), theGraph(graph)
		{
		}

		ContrastiveDivergence2D::~ContrastiveDivergence2D()
		{
		}
		
		void ContrastiveDivergence2D::optimize(int k, double betaJ, bool useLR, bool updateNorms, bool fixKappa, bool fixLambda)
		{
			auto theGraph = this->theGraph.lock();
			auto vis_0 = *((dynamic_pointer_cast<Storage>(theGraph->storages["visibles_pooled"].lock()))->storage[0]);
			auto spec_0 = *((dynamic_pointer_cast<Storage>(theGraph->storages["visibles_pooled"].lock()))->storage[0]);
			auto hid_0 = *((dynamic_pointer_cast<Storage>(theGraph->storages["hiddens_raw"].lock()))->storage[0]);
			auto vis_n = *((dynamic_pointer_cast<Storage>(theGraph->storages["visibles_raw"].lock()))->storage[k]);
			auto hid_n = *((dynamic_pointer_cast<Storage>(theGraph->storages["hiddens_raw"].lock()))->storage[k]);

			auto visDimx = vis_0.dimensions[0];
			auto hidDimx = hid_0.dimensions[0];
			auto hidDimy = hid_0.dimensions[1];
			auto samples = vis_0.dimensions.size() > 2 ? vis_0.dimensions[2] : 1;

			double delta = 0;
			int counter = 0;
			//auto positive = vis_0[{2*random_connection}] * hid_0[{random_connection}];
			//auto negative = vis_n[{2*random_connection}] * hid_n[{random_connection}];

			//delta += learningRate * ( positive-negative );	
			double test = 0;
			bool isCont = theGraph->variables.size() > 2 ? true : false;

			double exp_vis0 = 0;
			double exp_visn = 0;
			double vishid0 = 0;
			double vishidn = 0;
			double exp_hid0 = 0;
			double exp_hidn = 0;

#pragma omp parallel for reduction(+:delta, vishid0, vishidn,exp_vis0,exp_visn,exp_hid0,exp_hidn)
			for (int s = 0; s < samples; s++) {
#pragma omp parallel for reduction(+:delta,vishid0, vishidn,exp_vis0,exp_visn,exp_hid0,exp_hidn)
				for (int i = 0; i < hidDimx; i++) {
#pragma omp parallel for reduction(+:delta,vishid0, vishidn,exp_vis0,exp_visn,exp_hid0,exp_hidn)
					for (int j = 0; j < hidDimy; j++) {
						//learning data
						auto vc_0 = vis_0[{2*i,2*j,s}];
						auto vu_0 = vis_0[{2*i, 2*j +2* 1, s}];
						auto vd_0 = vis_0[{2*i, 2*j - 2*1, s}];
						auto vl_0 = vis_0[{2*i - 2*1, 2*j, s}];
						auto vr_0 = vis_0[{2*i + 2*1, 2*j, s}];
						//modeled data
						auto vc_n = vis_n[{2*i, 2*j, s}];
						auto vu_n = vis_n[{2*i, 2*j + 2*1, s}];
						auto vd_n = vis_n[{2*i, 2*j - 2*1, s}];
						auto vl_n = vis_n[{2*i - 2*1, 2*j, s}];
						auto vr_n = vis_n[{2*i + 2*1, 2*j, s}];
						//hidden
						auto hc_0 = hid_0[{i, j, s}];
						auto hc_n = hid_n[{i, j, s}];
						//correlations
						double correlation_0 = (1.0 / 4.0) * (vc_0*vu_0 + vc_0 * vd_0 + vc_0 * vl_0 + vc_0 * vr_0);
						double correlation_n = (1.0 / 4.0) * (vc_n*vu_n + vc_n * vd_n + vc_n * vl_n + vc_n * vr_n);
						delta += correlation_0 - correlation_n;
						vishid0 += correlation_0;
						vishidn += correlation_n;
						if (isCont) {
							exp_vis0 += pow(pow(vc_0, 2) - 1, 2);
							exp_visn += pow(pow(vc_n, 2) - 1, 2);
							exp_hid0 += pow(pow(hc_0, 2) - 1, 2);
							exp_hidn += pow(pow(hc_n, 2) - 1, 2);
						}
					}
				}
			}
			//take the average
			vishidn /= hidDimx *hidDimy* samples;
			vishid0 /= hidDimx * hidDimy* samples;
			exp_vis0 /= hidDimx * hidDimy* samples;
			exp_visn /= hidDimx * hidDimy* samples;
			delta /= hidDimx * hidDimy* samples;
			auto kappa = theGraph->getVarForName("kappa");
			auto Ah = theGraph->getVarForName("Ah");
			auto Av = theGraph->getVarForName("Av");
			auto lambda = theGraph->getVarForName("lambda");
			delta = vishid0 - vishidn;
			auto tmpDelta = abs(delta) > 0.2 ? (signbit(delta) ? -0.2 : 0.2) : delta;
			auto tmpVisDelta = abs(exp_vis0 - exp_visn) > 0.2 ? (signbit(exp_vis0 - exp_visn) ? -0.2 : 0.2) : exp_vis0 - exp_visn;
			auto tmpHidDelta = abs(exp_hid0 - exp_hidn) > 0.2 ? (signbit(exp_hid0 - exp_hidn) ? -0.2 : 0.2) : exp_hid0 - exp_hidn;
			if (!fixKappa && !isnan(tmpDelta))
				*kappa->value = *kappa->value + Tensor({ 1 }, { learningRate *(tmpDelta) });
			auto newValue = *Av->value + Tensor({ 1 }, { learningRate * (tmpVisDelta), 0.2 });
			auto newValue2 = *Ah->value + Tensor({ 1 }, { learningRate * (tmpHidDelta) });
			if (updateNorms && !(isnan(newValue) || isnan(newValue2))) {
				*Av->value = newValue;
				*Ah->value = newValue2;
			}

			if (lambda != nullptr) {
				auto tmpLamDelta = abs(exp_vis0 - exp_visn)>0.2 ? (signbit(exp_vis0 - exp_visn) ? -0.2 : 0.2) : exp_vis0 - exp_visn;//abs(exp_vis0 - exp_visn) > 0.2? (signbit(exp_vis0 - exp_visn) ? -0.2 : 0.2) : exp_vis0-exp_visn; //abs((exp_vis0 - 1)*(exp_vis0 - 1) - (exp_visn - 1)*(exp_visn - 1)) > 0.2 ? (signbit((exp_vis0 - 1)*(exp_vis0 - 1) - (exp_visn - 1)*(exp_visn - 1)) ? -0.2 : 0.2) : (exp_vis0 - 1)*(exp_vis0 - 1) - (exp_visn - 1)*(exp_visn - 1);
																																	//std::cout << tmpLamDelta << std::endl;
				if (!fixLambda && !isnan(tmpLamDelta))
					*lambda->value = *lambda->value + Tensor({ 1 }, { -learningRate * (tmpLamDelta) });
			}

			lambda.reset();
			kappa.reset();
			Av.reset();
			Ah.reset();
		}
	}
}
