#include "OptPlaceholder.h"

namespace ct {

	OptPlaceholder::OptPlaceholder(string name) : name(name)
	{
	}


	OptPlaceholder::~OptPlaceholder()
	{
	}

	shared_ptr<Tensor> ct::OptPlaceholder::compute(std::initializer_list<shared_ptr<Tensor>> input)
	{
		return shared_ptr<Tensor>();
	}

	shared_ptr<Tensor> ct::OptPlaceholder::compute(std::vector<shared_ptr<Tensor>> input)
	{
		return shared_ptr<Tensor>();
	}

	string ct::OptPlaceholder::type()
	{
		return "optplaceholder";
	}
}
