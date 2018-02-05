#pragma once
#include "Symmetry.h"
template<class T> class TranslationSymmetry :
	public Symmetry<T>
{
public:
	TranslationSymmetry<T>();
	~TranslationSymmetry<T>();
	int operator()(T input[], T output[], size_t length);
	//Symmetry<T> operator*(Symmetry<T> l);
};

