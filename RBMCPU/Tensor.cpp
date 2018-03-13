#include "Tensor.h"
#include <iostream>

namespace ct {

	Tensor::Tensor(std::initializer_list<int> dimensions) : dimensions(dimensions.begin(), dimensions.end())
	{
		int alldims = 1;
		for (auto dim : dimensions) {
			alldims *= dim;
		}
		elements = vector<double>(alldims);
		size = alldims;
	}

	Tensor::Tensor(std::initializer_list<int> dimensions, std::initializer_list<double> values) : dimensions(dimensions.begin(), dimensions.end()), elements(values.begin(), values.end())
	{
		int alldims = 1;
		for (auto dim : dimensions) {
			alldims *= dim;
		}
		size = alldims;
	}

	Tensor::Tensor(vector<int> dimensions) : dimensions(dimensions.begin(), dimensions.end())
	{
		int alldims = 1;
		for (auto dim : dimensions) {
			alldims *= dim;
		}
		size = alldims;
	}

	Tensor::Tensor(vector<int> dimensions, vector<double> values) : dimensions(dimensions.begin(), dimensions.end()),
		elements(values.begin(), values.end())
	{
		int alldims = 1;
		for (auto dim : dimensions) {
			alldims *= dim;
		}
		size = alldims;
	}

	Tensor::Tensor()
	{
		size = 0;
	}

	Tensor::~Tensor()
	{
	}
	double & Tensor::operator[](std::initializer_list<int> list)
	{
		// TODO: insert return statement here
		vector<int> tuple(list.begin(), list.end());
		int dim = tuple[0];
		int lastDimensions = dimensions[0];
		for (int i = 1; i < tuple.size(); i++) {
			if (tuple[i] > dimensions[i]) throw exception("Dimension do not match");
			dim += tuple[i] * lastDimensions;
			lastDimensions *= dimensions[i];
		}
		return elements[dim];
	}
	Tensor::operator double&()
	{
		if (dimensions.size() == 1 && dimensions[0] == 1) {
			return elements[0];
		}
		else {
			throw exception("Cannot cast tensor to scalar");
		}
	}
	Tensor Tensor::operator+(Tensor a)
	{
		Tensor newT(this->dimensions, this->elements);
		for (int i = 0; i < elements.size(); i++) {
			newT.elements[i] += a.elements[i];
		}
		return newT;
	}

	 Tensor Tensor::elementWise(std::function<double(double)> lambda)
	{
		for (int i = 0; i < size; i++) {
			this->elements[i] = lambda(this->elements[i]);
		}
		auto t = Tensor(*this);
		return t;
	}
	
}