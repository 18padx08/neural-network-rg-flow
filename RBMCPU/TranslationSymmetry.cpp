#include "TranslationSymmetry.h"
#include "SymmetryCombination.h"

template<class T>
int TranslationSymmetry<T>::operator()(T input[], T output[], size_t length)
{
	//move every element one place
	for (int i = 0; i < length; i++) {
		if (i < length-1) {
			output[i] = input[i + 1];
		}
		else if (i == length - 1) {
			//last element
			output[i] = input[0];
		}
	}
	return 0;
}

template<class T>
Symmetry<T> *TranslationSymmetry<T>::operator*(Symmetry<T> *l)
{
	SymmetryCombination<T> *newSymmetry = new SymmetryCombination<T>();
	newSymmetry->putSymmetry(l);
	newSymmetry->putSymmetry(this);
	return (Symmetry<T> *)newSymmetry;
}

