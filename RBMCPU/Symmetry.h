#pragma once

template<class T> 
class Symmetry
{

public:
	 virtual int operator()(T input[], T output[], size_t length) = 0;
	 //virtual Symmetry<T> operator*(Symmetry<T> l) = 0;
};

