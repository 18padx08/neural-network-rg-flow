#include "Graph.h"

namespace ct {
	void Graph::insert_nodes(shared_ptr<Node> parent, vector<shared_ptr<Node>> inputs)
	{
		for (auto input : inputs) {
			input->consumers.push_back(parent);
			flat_tree.insert(flat_tree.begin(), input);
			if (input->type() == "operation") {
				operations.insert(operations.begin(), input);
				insert_nodes(input,input->inputs);
			}
			else if (input->type() == "variable") {
				variables.insert(variables.begin(), input);
			}
			else if (input->type() == "placeholder") {
				placeholders.insert(placeholders.begin(), input);
			}
			else if (input->type() == "storage") {
				storages.insert(storages.begin(), input);
				insert_nodes(input, input->inputs);
			}
			else if (input->type() == "optplaceholder") {
				optplaceholders.insert(optplaceholders.begin(), input);
			}
		}
		//close graph if there is a optplaceholder
		if (optplaceholders.size() > 0) {
			optplaceholders[0]->inputs.push_back(flat_tree[flat_tree.size() - 1]);
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
			storages.insert(storages.begin(), begin);
			insert_nodes(begin, begin->inputs);
		}
		else if (begin->type() == "optplaceholder") {
			optplaceholders.insert(optplaceholders.begin(), begin);
		}

		
	}

	Graph::~Graph()
	{
	}
	void Graph::run()
	{
		
	}
}
