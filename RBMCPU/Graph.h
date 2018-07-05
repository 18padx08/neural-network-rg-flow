#pragma once
#include <vector>
#include <map>
#include <memory>
#include "Node.h"
#include "Variable.h"
#include "Operation.h"
#include "Storage.h"
#include "Placeholder.h"
#include "OptPlaceholder.h"

using namespace std;
namespace ct {
	
	class Graph
	{
	private:
		bool isClosed;
		int loopCounter = -1;
		vector<weak_ptr<Node>> operations;
		
		vector<weak_ptr<Node>> placeholders;
		
		void insert_nodes(weak_ptr<Node> parent, vector<weak_ptr<Node>> inputs);
		
	public:
		shared_ptr<Variable> getVarForName(string name);
		shared_ptr<Operation> getOperationForType(string name);
		vector<weak_ptr<Node>> optplaceholders;
		map<string,weak_ptr<Node>> storages;
		vector<weak_ptr<Node>> variables;
		vector<shared_ptr<Node>> flat_tree;
		const weak_ptr<Node> begin;
		weak_ptr<Node> currentNode;
		Graph(shared_ptr<Node> begin);
		~Graph();
		void run();
	};
}
