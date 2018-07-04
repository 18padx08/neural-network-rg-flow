#include "Variable.h"

namespace ct {
	Variable::Variable()
	{
	}


	Variable::~Variable()
	{
		inputs.clear();
		consumers.clear();
	}
	shared_ptr<Tensor> Variable::compute(std::initializer_list<weak_ptr<Tensor>> input)
	{
		vector<weak_ptr<Tensor>> tmp(input.begin(), input.end());
		return compute(tmp);
	}
	string Variable::type()
	{
		return "variable";
	}
	shared_ptr<Tensor> Variable::compute(std::vector<weak_ptr<Tensor>> input)
	{
		return shared_ptr<Tensor>();
	}
}
