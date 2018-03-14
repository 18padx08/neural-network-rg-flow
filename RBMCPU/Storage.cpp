#include "Storage.h"

namespace ct {
	Storage::Storage(shared_ptr<Node> input)
	{
		this->inputs.push_back(input);
	}


	Storage::~Storage()
	{
	}
	shared_ptr<Tensor> Storage::compute(std::initializer_list<shared_ptr<Tensor>> input)
	{
		return make_shared<Tensor>();
	}
	string Storage::type()
	{
		return "storage";
	}
	shared_ptr<Tensor> Storage::compute(std::vector<shared_ptr<Tensor>> input)
	{
		return shared_ptr<Tensor>();
	}
}
