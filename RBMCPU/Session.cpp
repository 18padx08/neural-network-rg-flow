#include "Session.h"

namespace ct {
	Session::Session(shared_ptr<Graph> graph) : graph(graph)
	{
		
	}

	Session::~Session()
	{
		
	}

	void Session::run()
	{
		do {
			//gather input
			if (graph->currentNode->type() == "placeholder") {
				auto castNode = dynamic_pointer_cast<Placeholder>(graph->currentNode);
				castNode->output = make_shared<Tensor>(Tensor({ 1 }, { feedDict[castNode->name] }));
			}
			else if (graph->currentNode->type() == "operation") {
				vector<shared_ptr<Tensor>> inputs;
				for (auto t : graph->currentNode->inputs) {
					inputs.push_back(t->output);
				}
				graph->currentNode->output = graph->currentNode->compute(inputs);
			}
			else if (graph->currentNode->type() == "storage") {

			}
			else if (graph->currentNode->type() == "variable") {
				auto castNode = dynamic_pointer_cast<Variable>(graph->currentNode);
				//dimension check
				castNode->output = castNode->value;
			}
			graph->currentNode = graph->currentNode->consumers[0];
		} while (graph->currentNode != graph->begin && graph->currentNode->consumers.size() > 0);
		cachedOutput = graph->currentNode->output;
	}
	void Session::run(map<string, double> f)
	{
		this->feedDict = f;
		run();
	}
}
