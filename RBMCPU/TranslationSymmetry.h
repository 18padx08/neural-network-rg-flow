#pragma once
#include "Symmetry.h"
template<class T> class TranslationSymmetry :
	public Symmetry<T>
{
private:
	int translationStep = 1;
public:
	TranslationSymmetry<T>() ;
	TranslationSymmetry<T>(int translationStep) ;
	int operator()(T input[], T output[], size_t length);
	vector<T> operator()(vector<T> &input);
	Symmetry<T> *operator*(Symmetry<T> *l);
};

