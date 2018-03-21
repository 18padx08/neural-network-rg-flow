#include "RGFlowTest.h"

using namespace ct;

RGFlowTest::RGFlowTest()
{
}


RGFlowTest::~RGFlowTest()
{
}

void RGFlowTest::run(double beta)
{
	//first layer setup
	auto graph = RBMCompTree::getRBMGraph();
	auto session = make_shared<Session>(Session(graph));
	Ising1D ising(500, beta, 1);
	vector<int> dims = { 500 };
	vector<double> values(500 * 50);
	for (int i = 0; i < 20000; i++) {
		ising.monteCarloStep();
	}
	auto conf = ising.getConfiguration();
	for (int i = 0; i < conf.size(); i++) {
		values[i] = conf[i] <= 0 ? -1 : 1;
	}

	map<string, shared_ptr<Tensor>> feedDic = { { "x", make_shared<Tensor>(Tensor(dims,values)) } };
	auto var = dynamic_pointer_cast<Variable>(graph->variables[0]);
	double average = 0;
	int counter = 1;
	auto correlationNN = ising.calcExpectationValue(1);
	auto correlationNNN = ising.calcExpectationValue(2);
	optimizers::ContrastiveDivergence cd(graph, 0.1, 0);
	dims = { 500,50 };
	std::ofstream of("coupling_traj_" + std::to_string(beta) + ".csv");
	for (int i = 0; i < 2500; i++) {
		for (int j = 0; j < 1; j++) {
			session->run(feedDic, true, 1);
			cd.optimize(1, correlationNNN);
		}
		std::cout << "\r" << "                                                                                                         ";
		std::cout << "\r" << "Coupling: " << (double)*(var->value) << " (avg. " << average / counter << ")";
		of << (double)*(var->value) << std::endl;
		if (i > 2000) {
			average += std::abs((double)*(var->value));
			std::cout << "\r" << "                                                                                                         ";
			std::cout << "\r" << "Coupling: " << (double)*(var->value) << " (avg. " << average / counter << ")";
			counter++;
		}
		for (int s = 0; s < 50; s++) {
			for (int i = 0; i < 1500; i++) {
				ising.monteCarloStep();
			}
			auto tmpconf = ising.getConfiguration();

			for (int i = 0; i < tmpconf.size(); i++) {
				values[i + s * 500] = tmpconf[i] <= 0 ? -1 : 1;
			}
		}
		auto t = make_shared<Tensor>(Tensor(dims, values));
		feedDic = { { "x", t } };
	}
	of.flush();
	var->value = make_shared<Tensor>(Tensor({ 1 }, { average / counter }));


	int val = 0;
	std::uniform_int_distribution<int> dist(0, 249);
	auto engine = std::default_random_engine(time(NULL));
	auto castNode = dynamic_pointer_cast<Storage>(graph->storages["hiddens_pooled"]);
	for (int i = 0; i < 5000; i++) {
		auto index = dist(engine);
		session->run(feedDic, true, 5);
		auto size = (*castNode->storage[4]).dimensions[0];
		auto val1 = (*castNode->storage[4])[{index}];
		auto val2 = (*castNode->storage[4])[{index + 1 < size ? index + 1 : 0}];
		val += val1 * val2;

		for (int i = 0; i < 1500; i++) {
			ising.monteCarloStep();
		}
		auto tmpconf = ising.getConfiguration();

		for (int i = 0; i < tmpconf.size(); i++) {
			values[i] = tmpconf[i] <= 0 ? -1 : 1;
		}

		feedDic = { { "x", make_shared<Tensor>(Tensor(dims,values)) } };
	}
	auto corrNNN = (double)val / 5000;

	std::cout << "Monte Carlo measurements vs. NN" << std::endl;
	std::cout << "<v_i v_{i+1}> ~ " << correlationNN << "  exact => " << tanh(beta * 1) << " error: " << abs(correlationNN - tanh(beta)) << std::endl;
	std::cout << "<v_i v_{i+2}> ~ " << correlationNNN << " exact => " << pow(tanh(beta * 1), 2) << " error: " << abs(correlationNNN - pow(tanh(beta), 2)) << std::endl;
	std::cout << "<h_i h_{i+1}> = " << corrNNN << " error: " << abs(corrNNN - pow(tanh(beta), 2)) << std::endl;
}
