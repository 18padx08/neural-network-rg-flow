#include "Storage.h"

namespace ct {
	Storage::Storage(weak_ptr<Node> input, string name) : name(name)
	{
		this->inputs.push_back(input);
	}


	Storage::~Storage()
	{
		inputs.clear();
		storage.clear();
	}
	shared_ptr<Tensor> Storage::compute(std::initializer_list<weak_ptr<Tensor>> input)
	{
		vector<weak_ptr<Tensor>> tmp(input.begin(), input.end());
		return compute(tmp);
	}
	string Storage::type()
	{
		return "storage";
	}
	shared_ptr<Tensor> Storage::compute(std::vector<weak_ptr<Tensor>> input)
	{
		return shared_ptr<Tensor>();
	}
}
