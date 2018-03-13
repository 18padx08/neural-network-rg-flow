#include "Placeholder.h"

namespace ct {

	Placeholder::Placeholder(std::string name) : name(name)
	{

	}


	Placeholder::~Placeholder()
	{
	}
	shared_ptr<Tensor> Placeholder::compute(std::initializer_list<shared_ptr<Tensor>> input)
	{
		return make_shared<Tensor>();
	}
	string Placeholder::type()
	{
		return "placeholder";
	}
	shared_ptr<Tensor> Placeholder::compute(std::vector<shared_ptr<Tensor>> input)
	{
		return shared_ptr<Tensor>();
	}
}
