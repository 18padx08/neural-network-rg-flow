#include "Add.h"


namespace ct {
	Add::Add()
	{
	}


	Add::~Add()
	{
	}

	shared_ptr<Tensor> ct::Add::compute(std::initializer_list<shared_ptr<Tensor>> input)
	{
		vector<shared_ptr<Tensor>> in(input.begin(), input.end());
		return compute(in);
	}

	shared_ptr<Tensor> ct::Add::compute(std::vector<shared_ptr<Tensor>> input)
	{
		return std::make_shared<Tensor>(*input[0] + *input[1]);
	}
}
