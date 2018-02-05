#pragma once
#include <vector>
#include "Symmetry.h"
template <class T>
class SymmetryCombination :
	public Symmetry<T>
{
private:
	std::vector<Symmetry<T>> symList;
public:
	 int operator()(T input[], T output[], size_t length) ;
	// Symmetry<T> operator*(Symmetry<T> l);
};


