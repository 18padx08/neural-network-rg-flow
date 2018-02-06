#pragma once
#include "Symmetry.h"
template<class T>
class Z2 :
	public Symmetry<T>
{
public:
	int operator()(T input[], T output[], size_t length);
	Symmetry<T> *operator*(Symmetry<T> *l);
};

