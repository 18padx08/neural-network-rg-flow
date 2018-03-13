#pragma once
#include <vector>
#include <map>
#include <memory>
#include "Graph.h"
#include "OptPlaceholder.h"
using namespace std;
namespace ct {
	class Session
	{
	private:
		map<string, shared_ptr<Tensor>> feedDict;
		shared_ptr<Graph> graph;
	public:
		Session(shared_ptr<Graph> graph);
		~Session();
		void run(bool isClosed = false, int runs = 1);
		void run(map<string, shared_ptr<Tensor>> f, bool isClosed = false, int runs =1);
		shared_ptr<Tensor> cachedOutput;
	};
}
