#pragma once
#include <vector>
using namespace std;
namespace ct {
	class Tensor
	{
		vector<double> elements;
	public:
		Tensor(std::initializer_list<int> dimensions);
		Tensor(std::initializer_list<int> dimensions, std::initializer_list<double> values);
		Tensor(vector<int> dimensions);
		Tensor(vector<int> dimensions, vector<double> values);
		Tensor();
		~Tensor();
		vector<int> dimensions;
		
		double& operator[](std::initializer_list<int> list);
		operator double&();

		//tensor operations
		Tensor operator+(Tensor a);
	};
	
}
