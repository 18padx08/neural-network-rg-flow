#pragma once
#include<vector>
using namespace std;
template<class T> 
class Symmetry
{

public:
	 virtual int operator()(T input[], T output[], size_t length) = 0;
	 virtual vector<T> operator()(vector<T> &input) = 0;
	 virtual Symmetry<T> *operator*(Symmetry<T> *l) = 0;
};

