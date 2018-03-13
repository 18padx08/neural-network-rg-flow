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
		virtual shared_ptr<Tensor> compute(std::initializer_list<shared_ptr<Tensor>> input) = 0;
		virtual shared_ptr<Tensor> compute(std::vector<shared_ptr<Tensor>> input) = 0;
		virtual string type() = 0;
		vector<shared_ptr<Node>> inputs;
		vector<shared_ptr<Node>> consumers;
		shared_ptr<Tensor> output;
	};
}
