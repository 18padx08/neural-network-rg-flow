#include "OptPlaceholder.h"

namespace ct {

	OptPlaceholder::OptPlaceholder(string name) : name(name)
	{
	}


	OptPlaceholder::~OptPlaceholder()
	{
		inputs.clear();
		consumers.clear();
	}

	shared_ptr<Tensor> ct::OptPlaceholder::compute(std::initializer_list<weak_ptr<Tensor>> input)
	{
		return shared_ptr<Tensor>();
	}

	shared_ptr<Tensor> ct::OptPlaceholder::compute(std::vector<weak_ptr<Tensor>> input)
	{
		return shared_ptr<Tensor>();
	}

	string ct::OptPlaceholder::type()
	{
		return "optplaceholder";
	}
}
