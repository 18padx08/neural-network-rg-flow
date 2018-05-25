#include "RGFlowTest.h"
#include "Phi1D.h"
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
	shared_ptr<Graph> graph = RBMCompTree::getRBMGraph();
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
	shared_ptr<Graph> graph = RBMCompTree::getRBMGraph();
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

	for (int some = 0; some < 5; some++) {
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
	int batchSize = 10;
	int num_layers = 4;
	int num_steps = 1500;
	int max_iterations = 1000;
	int spinChainSize = 512;
	Ising1D ising(spinChainSize, startingBeta, 1);
	auto samples = vector<int>(spinChainSize * batchSize);
	auto graphList = vector<shared_ptr<Graph>>();
	auto sessions = vector<Session>();
	for (int i = 0; i < num_layers; i++) {
		//different layers
		shared_ptr<Graph> graph = RBMCompTree::getRBMGraph();
		auto castNode = dynamic_pointer_cast<Variable>(graph->variables[0]);
		castNode->value = make_shared<Tensor>(Tensor({ 1 }, { -1 }));
		auto session = Session(graph);
		graphList.push_back(graph);
		sessions.push_back(session);
	}
	vector<double> values(spinChainSize * batchSize);
	vector<int> dims = { spinChainSize,batchSize };
	for (int layer = 0; layer < graphList.size(); layer++) {
		//create a ising batch
		std::ofstream of("rgFlow_layer_" + std::to_string(layer) + "_bs=" + std::to_string(batchSize) + "_cs=" + std::to_string(spinChainSize) + ".csv");
		int counter = 1;
		double average = 0;
		std::cout << std::endl << " === Layer " << layer << " === " << std::endl;
		for (int iteration = 0; iteration < 4; iteration++)
		{
			for (int sample = 0; sample < batchSize; sample++)
			{
				for (int step = 0; step < num_steps; step++)
				{
					ising.monteCarloStep();
				}
				auto tmpconf = ising.getConfiguration();

				for (int i = 0; i < tmpconf.size(); i++) {
					values[i + spinChainSize * sample] = tmpconf[i] <= 0 ? -1 : 1;
				}
			}
			//if first batch thermalize
			map<string, shared_ptr<Tensor>> feedDic = { { "x", make_shared<Tensor>(dims, values) } };

			if (iteration == 0) {
				for (int therm = 0; therm < 500; therm++) {
					feedDic = { { "x", make_shared<Tensor>(dims, values) } };
					for (int i = 0; i <= layer; i++)
					{
						//go step by step through the layers
						if (i > 0) {
							auto vals = (dynamic_pointer_cast<Storage>(graphList[i - 1]->storages["hiddens_pooled"])->storage[1]);
							feedDic = { { "x", make_shared<Tensor>(*vals) } };
						}
						sessions[i].run(feedDic, true, 1);
						if (i == layer) {
							optimizers::ContrastiveDivergence cd(graphList[i], 0.1, 0);
							cd.optimize(1, 1.0);
							auto var = dynamic_pointer_cast<Variable>(graphList[i]->variables[0]);
							std::cout << "\r" << "                                                                                                         ";
							std::cout << "\r" << "Coupling: " << (double)*(var->value) / 2.0 << " (avg. " << average / counter / 2.0 << ")";
						}
					}
				}
				std::cout << "== Thermalizing finished ==" << std::endl;

			}
			else
			{
				for (int i = 0; i <= layer; i++)
				{
					//go step by step through the layers
					if (i > 0) {
						auto vals = (dynamic_pointer_cast<Storage>(graphList[i - 1]->storages["hiddens_pooled"])->storage[1]);

						feedDic = { {"x", make_shared<Tensor>(*vals)} };
					}
					sessions[i].run(feedDic, true, 1);
					if (i == layer) {
						optimizers::ContrastiveDivergence cd(graphList[i], 0.1, 0);
						cd.optimize(1, 1);
						auto var = dynamic_pointer_cast<Variable>(graphList[i]->variables[0]);
						std::cout << "\r" << "                                                                                                         ";
						std::cout << "\r" << "Coupling: " << (double)*(var->value) / 2.0 << " (avg. " << average / counter / 2.0 << ")";
						of << std::endl << (double)*(var->value) / 2.0 << std::endl;
					}
				}

			}
		}
		of.close();
	}
}

void RGFlowTest::plotRGFlowNew(double startingBeta, int batch_size)
{
	//initialize layers
	int batchSize = batch_size;
	int num_layers = 5;
	int num_steps = 1500;
	int max_iterations = 400;
	int spinChainSize = 512;
	Phi1D ising(spinChainSize, startingBeta,0,0,0);
	auto samples = vector<double>(spinChainSize * batchSize);
	vector<int> dims = { spinChainSize,batchSize };
	vector<int> thermDims = { spinChainSize,batchSize };
	auto graphList = vector<shared_ptr<Graph>>();
	auto sessions = vector<Session>();

	//the optimizers
	vector<optimizers::ContrastiveDivergence> cds;
	double oldBeta = startingBeta;
	for (int i = 0; i < num_layers; i++) {
		//different layers
		shared_ptr<Graph> graph = RBMCompTree::getRBMGraph();
		auto castNode = graph->getVarForName("kappa");
		auto newBeta = oldBeta * oldBeta / (1.0 - 2.0 *oldBeta *oldBeta);
		oldBeta = newBeta;
		castNode->value = make_shared<Tensor>(Tensor({ 1 }, { newBeta }));
		auto session = Session(graph);
		cds.push_back(optimizers::ContrastiveDivergence(graph, 0.01, 0));
		graphList.push_back(graph);
		sessions.push_back(session);
	}
	map<string, shared_ptr<Tensor>> feedDic;
	//thermalize montecarlo
	/*for (int i = 0; i < 25000; i++) {
		ising.monteCarloStep();
	}*/
	
	ising.fftUpdate();

	//create one batch of data
	vector<double> thermSamps(batchSize  * spinChainSize);
	for (int sam = 0; sam < batchSize; sam++) {
		auto t = ising.getConfiguration();
		for (int i = 0; i < t.size(); i++) {
			thermSamps[i + sam * t.size()] = t[i] <= 0 ? -1 : 1;
		}
		ising.fftUpdate();
	}
	//use this batch to thermalize through the layer
	feedDic = { { "x", make_shared<Tensor>(Tensor(thermDims, thermSamps)) } };

	int counter = 0;
	int events = 0;
	double gradient = 0;
	double average = 0;
	double lastAverage = 0;
	int loops = 0;
	for (int layer = 0; layer < num_layers; layer++) {
		bool run = true;
		auto var = graphList[layer]->getVarForName("kappa");//dynamic_pointer_cast<Variable>(graphList[layer]->variables[0]);
		counter = 0;
		lastAverage = 0;
		double lastEpsilon = 0;
		double epsilon = 0;
		int averageCount = 20 * pow(1, layer);
		do {
			if (counter % averageCount == 0 && counter != 0) {
				//check if average is smaller
				var->value = make_shared<Tensor>(Tensor({ 1 }, { average }));
				auto diff = abs(average - lastAverage);
				lastEpsilon = epsilon;
				epsilon = abs((1.0 / sqrt(batchSize)) * average);
				if ((diff < epsilon) || abs(average) < 10e-6)
				{
					lastAverage = 0;
					average = 0;
					counter = 0;
					loops = 0;
					//run = false;
					break;
				}
				lastAverage = average;
				average = 0;
				counter = 0;
				loops++;
			}
			if (layer > 0) {
				auto vals = (dynamic_pointer_cast<Storage>(graphList[layer - 1]->storages["hiddens_raw"])->storage[1]);
				feedDic = { { "x", make_shared<Tensor>(Tensor(*vals)) } };
			}
			sessions[layer].run(feedDic, true, 1);
			cds[layer].optimize(1, 1.0, true);
			//of << (double)*castNode->value << std::endl;
			std::cout << "\r" << "                                                                                ";
			std::cout << "\r" << "Layer: " << layer << " " << (double)*var->value;
			counter++;

			average += (double)*var->value / averageCount;

		} while (run);
	}
	std::cout << std::endl;
	std::cout << "== Thermalized ==" << std::endl;
	std::ofstream output("rg_flow_comb_betaj=" + std::to_string(startingBeta) + "_bs=" + std::to_string(batchSize) + "_cs=" + std::to_string(spinChainSize) + ".csv");
	for (int layer = 0; layer < num_layers; layer++) {
		for (int it = 0; it < max_iterations; it++) {
			//each iterations needs a new batch
			for (int sam = 0; sam < batchSize; sam++) {
				auto t = ising.getConfiguration();
				for (int i = 0; i < t.size(); i++) {
					samples[i + sam * t.size()] = t[i] <= 0 ? -1 : 1;
				}
				ising.fftUpdate();
			}
			//propagate the batch through the layer
			if (it % 10 == 0) {
				std::cout << "\r" << "                                                                                                                       ";
				std::cout << "\r" << "Coupling [" << it << " of " << max_iterations << "]: ";
			}

			feedDic = { { "x", make_shared<Tensor>(Tensor(dims, samples)) } };
			for (int i = 0; i <= layer; i++) {
				if (i > 0) {
					//if not the first layer take the output from the last layer
					auto vals = (dynamic_pointer_cast<Storage>(graphList[i - 1]->storages["hiddens_raw"])->storage[1]);
					feedDic = { { "x", make_shared<Tensor>(*vals) } };
				}
				sessions[i].run(feedDic, true, 1);
				if (i == layer) {
					cds[layer].optimize(1, 10, true);
				}
			}
			auto var = graphList[layer]->getVarForName("kappa");//dynamic_pointer_cast<Variable>(graphList[layer]->variables[0]);
			if (it % 10 == 0) {
				std::cout << " " << (double)*(var->value) << " ";
			}
			if (it > 200) {
				output << (double) *(var->value) << ",";
			}
		}
		output << std::endl;
	}
	output.close();
}

void RGFlowTest::modTest(double startingBeta)
{
	//first layer setup
	shared_ptr<Graph> graph = RBMCompTree::getRBMGraph();
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
	double lr = 0.8;
	std::ofstream of("coupling_traj_" + std::to_string(startingBeta) + "_modCD.csv");
	for (int i = 0; i < 2000; i++) {
		for (int j = 0; j < 1; j++) {
			session->run(feedDic, true, 1);
			cd.optimize(lr, 1);
		}
		std::cout << "\r" << "                                                                                                         ";
		std::cout << "\r" << "Coupling: " << (double)*(var->value) << " (avg. " << average / (counter > 0 ? counter : 1.0) << ")";
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

void RGFlowTest::testGibbsConvergence()
{
	std::uniform_int_distribution<int> dist(0, 250);
	auto engine = std::default_random_engine(time(NULL));

	shared_ptr<Graph> graph = RBMCompTree::getRBMGraph();
	auto session = make_shared<Session>(Session(graph));
	auto var = dynamic_pointer_cast<Variable>(graph->variables[0]);
	var->value = make_shared<Tensor>(Tensor({ 1 }, { -2 }));
	vector<double> values(500 * 500);
	vector<int> dims = { 500,500 };
#pragma omp parallel for
	for (int s = 0; s < 500; s++) {
#pragma omp parallel for
		for (int i = 0; i < 500; i++) {
			values[i + s * 500] = i % 2 == 0 ? -1 : 1;
		}
	}
	auto t = make_shared<Tensor>(Tensor(dims, values));
	map<string, shared_ptr<Tensor>> feedDic = { { "x", t } };
	double average = 0;
	for (int i = 0; i < 100; i++) {
		session->run(feedDic, true, i);
		auto castNode = dynamic_pointer_cast<Storage>(graph->storages["hiddens_pooled"]);
		double val = 0;
		for (int s = 0; s < 50; s++) {
			for (int j = 0; j < 500; j++) {
				auto index = dist(engine);
				auto size = (*castNode->storage[i]).dimensions[0];
				auto val1 = (*castNode->storage[i])[{index, j}];
				auto val2 = (*castNode->storage[i])[{index + 1 < size ? index + 1 : 0, j}];
				val += val1 * val2;
			}
		}
		val /= 500.0 * 50;
		average += val;

		std::cout << "after [" << i << "] gibbs steps (<h_i*h_{i+1}>): " << val << " (" << pow(tanh(1.0), 2) << ") with a error of: " << abs(val - pow(tanh(1.0), 2)) << std::endl;
	}
	std::cout << "---- Average ----" << std::endl;
	std::cout << average / 100 << std::endl;
}

void RGFlowTest::cheatTest(double startingBeta)
{
	//first layer setup
	shared_ptr<Graph> graph = RBMCompTree::getRBMGraph();
	auto session = make_shared<Session>(Session(graph));
	Ising1D ising(500, startingBeta, 1);
	vector<int> dims = { 500 };
	vector<double> values(500 * 500);
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
	optimizers::CheatCD cd(graph);
	dims = { 500,500 };
	double lr = 0.8;
	var->value = make_shared<Tensor>(Tensor({ 1 }, { 2 * atanh(sqrt(correlationNNN)) }));
	std::ofstream of("coupling_traj_" + std::to_string(startingBeta) + "_cheatCD.csv");
	for (int i = 0; i < 20; i++) {
		if (i > 0) {
			for (int j = 0; j < 1; j++) {
				session->run(feedDic, true, 1);
				cd.optimize();
			}
		}
		std::cout << "\r" << "                                                                                                         ";
		std::cout << "\r" << "Coupling: " << (double)*(var->value) << "[" << i << "]  (avg. " << average / (counter > 0 ? counter : 1.0) << ")";
		of << (double)*(var->value) << std::endl;
		if (i > 0) {
			counter++;
			average += std::abs((double)*(var->value));

			var->value = make_shared<Tensor>(Tensor({ 1 }, { average / counter }));

		}
		for (int s = 0; s < 500; s++) {
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
	for (int i = 0; i < 500; i++) {
		auto index = dist(engine);
		session->run(feedDic, true, 20);
		auto size = (*castNode->storage[20]).dimensions[0];
		auto val1 = (*castNode->storage[20])[{index}];
		auto val2 = (*castNode->storage[20])[{index + 1 < size ? index + 1 : 0}];
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
	auto corrNNN = (double)val / 500;

	std::cout << "Monte Carlo measurements vs. NN" << std::endl;
	std::cout << "<v_i v_{i+1}> ~ " << correlationNN << "  exact => " << tanh(startingBeta * 1) << " error: " << abs(correlationNN - tanh(startingBeta)) << std::endl;
	std::cout << "<v_i v_{i+2}> ~ " << correlationNNN << " exact => " << pow(tanh(startingBeta * 1), 2) << " error: " << abs(correlationNNN - pow(tanh(startingBeta), 2)) << std::endl;
	std::cout << "<h_i h_{i+1}> = " << corrNNN << " error: " << abs(corrNNN - pow(tanh(startingBeta), 2)) << std::endl;
}
