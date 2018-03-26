#include "RGFlowTest.h"

using namespace ct;

RGFlowTest::RGFlowTest()
{
}


RGFlowTest::~RGFlowTest()
{
}

void RGFlowTest::plotConvergence(double beta)
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
	for (int i = 0; i < 10000; i++) {
		for (int j = 0; j < 1; j++) {
			session->run(feedDic, true, 1);
			cd.optimize(1, correlationNNN);
		}
		std::cout << "\r" << "                                                                                                         ";
		std::cout << "\r" << "Coupling: " << (double)*(var->value) << " (avg. " << average / counter << ")";
		of << (double)*(var->value) << std::endl;
		if (i > 9000) {
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

void RGFlowTest::plotError(vector<int> num_samples)
{
	//first layer setup
	auto graph = RBMCompTree::getRBMGraph();
	auto session = make_shared<Session>(Session(graph));
	Ising1D ising(500, 1, 1);
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
	
	for (int i = 0; i < 100; i++) {
		for (int j = 0; j < 1; j++) {
			session->run(feedDic, true, 1);
			cd.optimize(1, correlationNNN);
		}
		std::cout << "\r" << "                                                                                                         ";
		std::cout << "\r" << "Coupling: " << (double)*(var->value) << " (avg. " << average / counter << ")";
		
		if (i > 50) {
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
	
	var->value = make_shared<Tensor>(Tensor({ 1 }, { average / counter }));


	int val = 0;
	std::uniform_int_distribution<int> dist(0, 249);
	auto engine = std::default_random_engine(time(NULL));
	auto castNode = dynamic_pointer_cast<Storage>(graph->storages["hiddens_pooled"]);

	for(int some =0; some < 5; some++) {
		for (int s = 0; s < num_samples.size(); s++) {
			std::cout << "---" << std::to_string(num_samples[s]) << "---" << std::endl;
			for (int i = 0; i < num_samples[s]; i++) {
				auto index = dist(engine);
				session->run(feedDic, true, 50);
				auto size = (*castNode->storage[50]).dimensions[0];
				auto val1 = (*castNode->storage[50])[{index}];
				auto val2 = (*castNode->storage[50])[{index + 1 < size ? index + 1 : index - 1}];

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
			auto corrNNN = (double)val / num_samples[s];
			std::ofstream error("neural_net_error.csv");
			error << abs(corrNNN - pow(tanh(1), 2)) << std::endl;

			std::cout << "Monte Carlo measurements vs. NN" << std::endl;
			std::cout << "<v_i v_{i+1}> ~ " << correlationNN << "  exact => " << tanh(1 * 1) << " error: " << abs(correlationNN - tanh(1)) << std::endl;
			std::cout << "<v_i v_{i+2}> ~ " << correlationNNN << " exact => " << pow(tanh(1 * 1), 2) << " error: " << abs(correlationNNN - pow(tanh(1), 2)) << std::endl;
			std::cout << "<h_i h_{i+1}> = " << corrNNN << " error: " << abs(corrNNN - pow(tanh(1), 2)) << std::endl;
			val = 0;
		}
	}
	
}

void RGFlowTest::plotRGFlow(double startingBeta)
{
	int batchSize = 50;
	int num_layers = 4;
	int num_steps = 1500;
	int max_iterations = 1000;
	Ising1D ising(500, startingBeta, 1);
	auto samples = vector<int>(500 * 50);
	auto graphList = vector<shared_ptr<Graph>>();
	auto sessions = vector<Session>();
	for (int i = 0; i < num_layers; i++) {
		//different layers
		auto graph = RBMCompTree::getRBMGraph();
		auto session = Session(graph);
		graphList.push_back(graph);
		sessions.push_back(session);
	}
	vector<double> values(500 * 50);
	vector<int> dims = { 500,50 };
	for (int layer = 0; layer < graphList.size(); layer++) {
		//create a ising batch
		int counter = 1;
		double average = 0;
		std::cout << std::endl << " === Layer " << layer << " === " <<std::endl;
		for (int iteration = 0; iteration < max_iterations *(layer+1); iteration++) 
		{
			for (int sample = 0; sample < batchSize; sample++) 
			{
				for (int step = 0; step < num_steps; step++) 
				{
					ising.monteCarloStep();
				}
				auto tmpconf = ising.getConfiguration();

				for (int i = 0; i < tmpconf.size(); i++) {
					values[i + 500*sample] = tmpconf[i] <= 0 ? -1 : 1;
				}
			}
			map<string, shared_ptr<Tensor>> feedDic = { {"x", make_shared<Tensor>(dims, values)} };
			for (int i = 0; i <= layer; i++) 
			{
				//go step by step through the layers
				if (i > 0) {
					auto vals = (dynamic_pointer_cast<Storage>(graphList[i - 1]->storages["hiddens_pooled"])->storage[9]);
					
					feedDic = { {"x", make_shared<Tensor>(*vals)} };
				}
				sessions[i].run(feedDic, true, 10);
				if (i == layer) {
					optimizers::ContrastiveDivergence cd(graphList[i], 0.1,0);
					cd.optimize(1, 10);
					auto var = dynamic_pointer_cast<Variable>(graphList[i]->variables[0]);
					std::cout << "\r" << "                                                                                                         ";
					std::cout << "\r" << "Coupling: " << (double)*(var->value)/2.0 << " (avg. " << average / counter/2.0 << ")";

					if (iteration > max_iterations*(layer+1) - 500) {
						average += std::abs((double)*(var->value));
						std::cout << "\r" << "                                                                                                         ";
						std::cout << "\r" << "Coupling: " << (double)*(var->value)/2.0 << " (avg. " << average / counter/2.0 << ")";
						counter++;
					}
				}
			}
		}
	}
}

void RGFlowTest::modTest(double startingBeta)
{
	//first layer setup
	auto graph = RBMCompTree::getRBMGraph();
	auto session = make_shared<Session>(Session(graph));
	Ising1D ising(500, startingBeta, 1);
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
	int counter = 0;
	auto correlationNN = ising.calcExpectationValue(1);
	auto correlationNNN = ising.calcExpectationValue(2);
	optimizers::ModCD cd(graph, 0.1, 0);
	dims = { 500,50 };
	double lr =0.8;
	std::ofstream of("coupling_traj_" + std::to_string(startingBeta) + "_modCD.csv");
	for (int i = 0; i < 2000; i++) {
		for (int j = 0; j < 1; j++) {
			session->run(feedDic, true, 1);
			cd.optimize(lr,1);
		}
		std::cout << "\r" << "                                                                                                         ";
		std::cout << "\r" << "Coupling: " << (double)*(var->value) << " (avg. " << average / (counter>0? counter:1.0) << ")";
		of << (double)*(var->value) << std::endl;
		if (i > 1) {
			counter++;
			average += std::abs((double)*(var->value));
			
			
			if (i % 150 == 0 && lr > 0.08) {
				var->value = make_shared<Tensor>(Tensor({ 1 }, { -average / counter }));
				average = 0;
				counter = 0;
				lr /= 2.0;
			}
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
		session->run(feedDic, true, 50);
		auto size = (*castNode->storage[50]).dimensions[0];
		auto val1 = (*castNode->storage[50])[{index}];
		auto val2 = (*castNode->storage[50])[{index + 1 < size ? index + 1 : 0}];
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
	std::cout << "<v_i v_{i+1}> ~ " << correlationNN << "  exact => " << tanh(startingBeta * 1) << " error: " << abs(correlationNN - tanh(startingBeta)) << std::endl;
	std::cout << "<v_i v_{i+2}> ~ " << correlationNNN << " exact => " << pow(tanh(startingBeta * 1), 2) << " error: " << abs(correlationNNN - pow(tanh(startingBeta), 2)) << std::endl;
	std::cout << "<h_i h_{i+1}> = " << corrNNN << " error: " << abs(corrNNN - pow(tanh(startingBeta), 2)) << std::endl;
}
