#include "Z2.h"


template<class T>
 int Z2<T>::operator()(T input[], T output[], size_t length)
{
	 for (int i = 0; i < length; i++) {
		 output[i] = input[i] <= 0 ? 1 : 0;
	 }
	return 0;
}

template<class T>
 Symmetry<T>* Z2<T>::operator*(Symmetry<T>* l)
{
	 SymmetryCombination<T> *newSymmetry = new SymmetryCombination<T>();
	 newSymmetry->putSymmetry(l);
	 newSymmetry->putSymmetry(this);
	 return (Symmetry<T> *)newSymmetry;
}

