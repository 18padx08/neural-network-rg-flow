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
			auto castNode = dynamic_pointer_cast<Storage>(store.second.lock());
			castNode->storage.clear();
		}
		(this->graph->optplaceholders[0].lock()->inputs[0].lock())->output = shared_ptr<Tensor>();
		do {
			if (i >= graph->flat_tree.size()) {
				i = 0;
				loopCounter++;
			}
			//don't relay on type but rather make dynamic_pointer_casts
			auto element = graph->flat_tree[i];
			auto optplaceholder = dynamic_pointer_cast<OptPlaceholder>(element);
			auto placeholder = dynamic_pointer_cast<Placeholder>(element);
			auto operation = dynamic_pointer_cast<Operation>(element);
			auto storage = dynamic_pointer_cast<Storage>(element);
			auto variable = dynamic_pointer_cast<Variable>(element);
			element.reset();
			//gather input
			if (optplaceholder != nullptr) {
				
				if (!optplaceholder->inputs[0].lock()->output ) {
					optplaceholder->output = make_shared<Tensor>(*feedDict[optplaceholder->name].lock());
				}
				else {
					auto tmpShared = optplaceholder->inputs[0].lock();
					auto outputcopy = make_shared<Tensor>(*tmpShared->output);
					optplaceholder->output = outputcopy;
				}
				optplaceholder.reset();
			}
			else if (placeholder != nullptr) {
					placeholder->output = make_shared<Tensor>(*feedDict[placeholder->name].lock());
					placeholder.reset();
			}
			else if (operation != nullptr) {
				vector<weak_ptr<Tensor>> inputs;
				for (auto t : operation->inputs) {
					auto tmp = make_shared<Tensor>(*t.lock()->output);
					inputs.push_back(tmp);
				}
				auto tmp = operation->output;
				auto output = operation->compute(inputs);
				
				operation->output = output;
				
			}
			else if (storage != nullptr) {
				auto storageValue = storage->inputs[0].lock();
				storage->storage.push_back(make_shared<Tensor>(*storageValue->output));
				storage->output = make_shared<Tensor>(*storageValue->output);
				storage.reset();
			}
			else if (variable != nullptr) {
				//dimension check
				variable->output = variable->value;
				variable.reset();
			}
			i++;
			//loopCounter++;
		} while (i < graph->flat_tree.size() || (isClosed && loopCounter < runs));

		//cachedOutput = make_shared<Tensor>(*graph->flat_tree[i-1]->output);
	}
	void Session::run(map<string, shared_ptr<Tensor>> f, bool isClosed, int runs)
	{
		this->feedDict.clear();
		for (auto & keyvalue : f) {
			this->feedDict.insert(keyvalue);
		}
		run(isClosed, runs);
	}
}
