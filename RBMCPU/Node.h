#pragma once
#include "Node.h"
#include <vector>
#include <memory>
#include "Tensor.h"
#include <string>

using namespace std;
namespace ct {
	class Node
	{
	public:
		Node();
		~Node();
		virtual
			shared_ptr<Tensor> compute(std::initializer_list<weak_ptr<Tensor>> input) = 0;
		virtual 
			shared_ptr<Tensor> compute(std::vector<weak_ptr<Tensor>> input) = 0;
		virtual 
			string type() = 0;
		vector<weak_ptr<Node>> inputs;
		vector<weak_ptr<Node>> consumers;
		shared_ptr<Tensor> output;
	};
}
