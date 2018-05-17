#pragma once
#include <vector>
#include <map>;
#include <memory>
#include "Node.h"
#include "Variable.h"
#include "Operation.h"
#include "Storage.h"
#include "Placeholder.h"

using namespace std;
namespace ct {
	
	class Graph
	{
	private:
		bool isClosed;
		int loopCounter = -1;
		vector<shared_ptr<Node>> operations;
		
		vector<shared_ptr<Node>> placeholders;
		
		void insert_nodes(shared_ptr<Node> parent, vector<shared_ptr<Node>> inputs);
		
	public:
		shared_ptr<Variable> getVarForName(string name);
		vector<shared_ptr<Node>> optplaceholders;
		map<string,shared_ptr<Node>> storages;
		vector<shared_ptr<Node>> variables;
		vector<shared_ptr<Node>> flat_tree;
		const shared_ptr<Node> begin;
		shared_ptr<Node> currentNode;
		Graph(shared_ptr<Node> begin);
		~Graph();
		void run();
	};
}
