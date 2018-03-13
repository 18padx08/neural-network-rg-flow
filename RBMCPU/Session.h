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
		map<string, double> feedDict;
		shared_ptr<Graph> graph;
	public:
		Session(shared_ptr<Graph> graph);
		~Session();
		void run();
		void run(map<string, double> f);
		shared_ptr<Tensor> cachedOutput;
	};
}
