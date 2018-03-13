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
	public:
		const shared_ptr<Node> begin;
		shared_ptr<Node> currentNode;
		Graph(shared_ptr<Node> begin);
		~Graph();
		void run();
	};
}
