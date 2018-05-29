#pragma once
#include <vector>
#include <functional>
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
		Tensor(Tensor& tensor);
		Tensor();
		~Tensor();
		vector<int> dimensions;
		int size;
		
		double& operator[](std::initializer_list<int> list);
		double*  getPointer(std::initializer_list<int> list);
		operator double&();
		void rescale(double factor);
		//tensor operations
		Tensor operator+(Tensor a);
		Tensor elementWise(std::function<double(double)> lambda);
		//template<typename F>
		//Tensor elementWise(F && lambda);
	};
	
	

}
