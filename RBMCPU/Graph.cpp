#include "Graph.h"

namespace ct {


	Graph::Graph(shared_ptr<Node> begin) : begin(begin)
	{
		currentNode = this->begin;
	}

	Graph::~Graph()
	{
	}
	void Graph::run()
	{
		
	}
}
