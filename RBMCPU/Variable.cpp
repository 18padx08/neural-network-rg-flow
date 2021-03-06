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
	shared_ptr<Tensor> Variable::compute(std::initializer_list<shared_ptr<Tensor>> input)
	{
		return make_shared<Tensor>();
	}
	string Variable::type()
	{
		return "variable";
	}
	shared_ptr<Tensor> Variable::compute(std::vector<shared_ptr<Tensor>> input)
	{
		return shared_ptr<Tensor>();
	}
}
