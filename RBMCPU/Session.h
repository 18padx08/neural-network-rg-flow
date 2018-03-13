#pragma once
#include <vector>
#include <map>
#include <memory>
#include "Graph.h"
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
		void run();
		void run(map<string, shared_ptr<Tensor>> f);
		shared_ptr<Tensor> cachedOutput;
	};
}
