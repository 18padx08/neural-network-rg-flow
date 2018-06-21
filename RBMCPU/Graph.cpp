#include "Graph.h"

namespace ct {
	shared_ptr<Variable> Graph::getVarForName(string name) {
		int i = 0;
		auto input = this->variables;
		for (auto && ele : input) {
			shared_ptr<Variable> tmp = dynamic_pointer_cast<Variable>(ele.lock());
			if (!tmp) continue;
			if (tmp->name == name) {
				return tmp;
			}
			i++;
		}
		return nullptr;
	}
	shared_ptr<Operation> Graph::getOperationForType(string name)
	{
		int i = 0; 
		auto input = this->operations;
		for (auto && ele : input) {
			shared_ptr<Operation> tmp = dynamic_pointer_cast<Operation>(ele.lock());
			if (!tmp) continue;
			if (tmp->type() == name) {
				return tmp;
			}
			i++;
		}
		return nullptr;
	}
	void Graph::insert_nodes(weak_ptr<Node> parent, vector<weak_ptr<Node>> inputs)
	{
		for (auto input : inputs) {
			auto newInput = input.lock();
			newInput->consumers.push_back(parent);
			flat_tree.insert(flat_tree.begin(), newInput);
			
			auto operation = dynamic_pointer_cast<Operation>(newInput);
			auto variable = dynamic_pointer_cast<Variable>(newInput);
			auto placeholder = dynamic_pointer_cast<Placeholder>(newInput);
			auto optplaceholder = dynamic_pointer_cast<OptPlaceholder> (newInput);
			auto storage = dynamic_pointer_cast<Storage> (newInput);
			if (operation != nullptr) {
				operations.insert(operations.begin(), newInput);
				insert_nodes(newInput, newInput->inputs);
			}
			else if (variable != nullptr) {
				variables.insert(variables.begin(), newInput);
			}
			else if (placeholder != nullptr) {
				placeholders.insert(placeholders.begin(), newInput);
			}
			else if (storage != nullptr) {
				auto castNode = dynamic_pointer_cast<Storage>(newInput);
				storages.insert(storages.begin(), { castNode->name, newInput });
				insert_nodes(newInput, newInput->inputs);
			}
			else if (optplaceholder != nullptr) {
				optplaceholders.insert(optplaceholders.begin(), newInput);
			}
		}
		
		
	}
	/// TODO rename begin to end if it works
	Graph::Graph(shared_ptr<Node> begin)
	{
		flat_tree.insert(flat_tree.begin(), begin);
		//build graph
		if (begin->type() == "operation") {
			operations.insert(operations.begin(), begin);
			insert_nodes(begin, begin->inputs);
		}
		else if (begin->type() == "variable") {
			variables.insert(variables.begin(), begin);
		}
		else if (begin->type() == "placeholder") {
			placeholders.insert(placeholders.begin(), begin);
		}
		else if (begin->type() == "storage") {
			auto castNode = dynamic_pointer_cast<Storage>(begin);
			storages.insert(storages.begin(), { castNode->name, begin });
			insert_nodes(begin, begin->inputs);
		}
		else if (begin->type() == "optplaceholder") {
			optplaceholders.insert(optplaceholders.begin(), begin);
		}
		//close graph if there is a optplaceholder
		if (optplaceholders.size() > 0) {
			auto weakptr = optplaceholders[0].lock();
			weakptr->inputs.push_back(flat_tree[flat_tree.size() - 1]);
		}
		
	}

	Graph::~Graph()
	{
		storages.clear();
		variables.clear();
		placeholders.clear();
		optplaceholders.clear();
		flat_tree.clear();
		currentNode.reset();
	}
	void Graph::run()
	{
		
	}
}
