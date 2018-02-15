#include "TIRBM.h"



TIRBM::~TIRBM()
{

}

void TIRBM::setSymmetries(vector<Symmetry<int>> symmetries)
{
	n_sym = symmetries.size();
	for (auto sym : symmetries) {
		symmetries.push_back(sym);
	}
}


void TIRBM::setParameters(ParamSet set)
{
}

void TIRBM::train(vector<vector<double>>& input, int sample_size, int epoch)
{
}



double * TIRBM::sample_from_net(int gibbs_steps)
{
	return nullptr;
}

double * TIRBM::reconstruct(vector<double>& input, int gibbs_steps)
{
	return nullptr;
}



void TIRBM::saveToFile(std::string filename)
{
}

void TIRBM::saveVisualization()
{
}

bool TIRBM::loadWeights(std::string filename)
{
	return false;
}

void TIRBM::propagate_down(vector<vector<double>>& input, vector<vector<double>>& output, int sample_size)
{
}

void TIRBM::propagate_up(vector<vector<double>>& input, vector<vector<double>>& output, int gibbs_steps)
{
}


double TIRBM::bernoulli(double p)
{
	return 0.0;
}

double TIRBM::uniform(double min, double max)
{
	return 0.0;
}

void TIRBM::sample_h_given_v(vector<int>& vis_src, vector<vector<double>>& hid_target, vector<int>& hid_target_sample, vector<int>& max_pooled_s)
{
	double constantPart = 0;
	vector<double> symPart(n_sym, 0);
	int num_hid = (int)n_hid;
	int num_vis = (int)n_vis;

	for (int i = 0; i < num_hid; i++) {
		for (int s = 0; s < n_sym; s++) {
			
			for (int j = 0; j < num_vis; j++) {
				//calculate the constant part only once
				if (i == 0) {
					constantPart += this->wij[j][i] * (symmetries[s](vis_src))[j];
				}
				symPart[s] += this->wij[j][i] * (symmetries[s](vis_src))[j];
			}
			symPart[s] += bjs[i][s];
			//again only compute the constant part once
			if (i == 0) {
				constantPart += bjs[i][s];
			}
			
		}
		for (int s = 0; s < n_sym; s++) {
			hid_target[i][s] = std::exp(symPart[s]) / (1 + std::exp(constantPart));
		}
		max_pooled_s[i] = max_pool(hid_target[i]);
		hid_target_sample[i] = hid_target[i][max_pooled_s[i]];
	}
}
//TODO implement symmetries as matrices
void TIRBM::sample_v_given_h(vector<int>& hid_src, vector<double>& vis_target, vector<int>& vis_target_sample, vector<int> &max_pooled_s)
{
	double pre_sigmoid = 0;
	int num_vis = (int)n_vis;
	int num_hid = (int)n_hid;

	for (int i = 0; i < num_vis; i++) {
		pre_sigmoid = 0;
		for (int s = 0; s < n_sym; s++) {
			for (int j = 0; j < num_hid; j++) {

				pre_sigmoid += (symmetries[s](this->wij[i]))[j] * hid_src[j];


				pre_sigmoid += bjs[i][s];
			}
		}

		vis_target[i] = actFun(pre_sigmoid);
		vis_target_sample[i] = bernoulli(vis_target[i]);
	}

}

double TIRBM::contrastive_divergence(vector<vector<int>>& input, int cdK, int batchSize)
{
#pragma parallel for
	for (int numBatch = 0; numBatch < batchSize; numBatch++) {
		vector<vector<double>> hid0(n_hid, vector<double>(n_sym));
		vector<int> hid0_sampled(n_hid);
		vector<vector<double>> hidN(n_hid, vector<double>(n_sym));
		vector<int> hidN_sampled(n_hid);
		vector<double> visN(n_vis);
		vector<int> visN_sampled(n_vis);
		vector<int> max_pooled_s(n_vis);
		//do CDk
		for (int i = 0; i < cdK; i++) {
			if (i == 0) {
				sample_h_given_v(input[numBatch], hid0, hid0_sampled, max_pooled_s);
				sample_v_given_h(hid0_sampled, visN, visN_sampled, max_pooled_s);
			}
			else {
				sample_h_given_v(visN_sampled, hidN, hidN_sampled, max_pooled_s);
				sample_v_given_h(hid0_sampled, visN, visN_sampled, max_pooled_s);
			}
		}
		//calculate the gradients for each batch
	}
	//apply the changes to the values
	return 0.0;
}

int TIRBM::max_pool(vector<double> hid_fixedj)
{
	return *std::max_element(hid_fixedj.begin(), hid_fixedj.end());
}



TIRBM::TIRBM(int n_vis, int n_hid, FunctionType activationFunction) :
	generator(time(NULL)),
	distribution(0.0,1.0), actFun(activationFunction),
	wij(n_vis,vector<double>(n_hid)),
	dWij(n_vis, vector<double>(n_hid)),
	tmpDwij(n_vis, vector<double>(n_hid)),
	bjs(n_hid,vector<double>()),
	dbjs(n_hid, vector<double>()),
	ci(n_vis),
	dci(n_vis),
	n_hid(n_hid),
	n_vis(n_vis),
	n_sym(0)
{


}
