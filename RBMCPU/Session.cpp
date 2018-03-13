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
		int i = 0;
		do {
			//gather input
			if (graph->flat_tree[i]->type() == "placeholder") {
				auto castNode = dynamic_pointer_cast<Placeholder>(graph->flat_tree[i]);
				castNode->output = feedDict[castNode->name];
			}
			else if (graph->flat_tree[i]->type() == "operation") {
				vector<shared_ptr<Tensor>> inputs;
				for (auto t : graph->flat_tree[i]->inputs) {
					inputs.push_back(t->output);
				}
				graph->flat_tree[i]->output = graph->flat_tree[i]->compute(inputs);
			}
			else if (graph->flat_tree[i]->type() == "storage") {

			}
			else if (graph->flat_tree[i]->type() == "variable") {
				auto castNode = dynamic_pointer_cast<Variable>(graph->flat_tree[i]);
				//dimension check
				castNode->output = castNode->value;
			}
			i++;
		} while (i < graph->flat_tree.size());
		cachedOutput = graph->flat_tree[i-1]->output;
	}
	void Session::run(map<string, shared_ptr<Tensor>> f)
	{
		this->feedDict = f;
		run();
	}
}
