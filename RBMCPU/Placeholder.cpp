#include "Placeholder.h"

namespace ct {

	Placeholder::Placeholder(std::string name) : name(name)
	{

	}


	Placeholder::~Placeholder()
	{
		inputs.clear();
		consumers.clear();
	}
	shared_ptr<Tensor> Placeholder::compute(std::initializer_list<weak_ptr<Tensor>> input)
	{
		vector<weak_ptr<Tensor>> tmp(input.begin(), input.end());
		return compute(tmp);
	}
	string Placeholder::type()
	{
		return "placeholder";
	}
	shared_ptr<Tensor> Placeholder::compute(std::vector<weak_ptr<Tensor>> input)
	{
		return shared_ptr<Tensor>();
	}
}
