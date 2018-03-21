#include "Session.h"

namespace ct {
	Session::Session(shared_ptr<Graph> graph) : graph(graph)
	{
		
	}

	Session::~Session()
	{
		
	}

	void Session::run(bool isClosed, int runs )
	{
		int i = 0;
		int loopCounter = 0;
		for (auto store : graph->storages) {
			auto castNode = dynamic_pointer_cast<Storage>(store.second);
			castNode->storage.clear();
		}
		this->graph->optplaceholders[0]->inputs[0]->output = shared_ptr<Tensor>();
		do {
			if (i >= graph->flat_tree.size()) {
				i = 0;
				loopCounter++;
			}
			//gather input
			if (graph->flat_tree[i]->type() == "optplaceholder") {
				auto castNode = dynamic_pointer_cast<OptPlaceholder>(graph->flat_tree[i]);
				if (!castNode->inputs[0]->output ) {
					castNode->output = feedDict[castNode->name];
				}
				else {
					castNode->output = castNode->inputs[0]->output;
				}
			}
			else if (graph->flat_tree[i]->type() == "placeholder") {
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
				auto castNode = dynamic_pointer_cast<Storage>(graph->flat_tree[i]);
				castNode->storage.push_back(castNode->inputs[0]->output);
				castNode->output = make_shared<Tensor>(*castNode->inputs[0]->output);
			}
			else if (graph->flat_tree[i]->type() == "variable") {
				auto castNode = dynamic_pointer_cast<Variable>(graph->flat_tree[i]);
				//dimension check
				castNode->output = castNode->value;
			}
			i++;
			//loopCounter++;
		} while (i < graph->flat_tree.size() || (isClosed && loopCounter < runs));
		cachedOutput = graph->flat_tree[i-1]->output;
	}
	void Session::run(map<string, shared_ptr<Tensor>> f, bool isClosed, int runs)
	{
		this->feedDict = f;
		run(isClosed, runs);
	}
}
