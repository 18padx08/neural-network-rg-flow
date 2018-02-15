#include "TranslationSymmetry.h"
#include "SymmetryCombination.h"

template<class T>
int TranslationSymmetry<T>::operator()(T input[], T output[], size_t length)
{
	//move every element one place
	for (int i = 0; i < length; i++) {
		if (i < length-translationStep) {
			output[i] = input[i + translationStep];
		}
		else if (i >= length - translationStep) {
			//calculate mod
			output[i] = input[(i+translationStep)%length];
		}
	}
	return 0;
}

template<class T>
vector<T> TranslationSymmetry<T>::operator()(vector<T>& input)
{
	vector<T> output(input.size());
	for (int i = 0; i < input.size(); i++) {
		if (i < length - translationStep) {
			output[i] = input[i + translationStep];
		}
		else if (i >= length - translationStep) {
			//calculate mod
			output[i] = input[(i + translationStep) % length];
		}
	}
	return output;
}

template<class T>
Symmetry<T> *TranslationSymmetry<T>::operator*(Symmetry<T> *l)
{
	SymmetryCombination<T> *newSymmetry = new SymmetryCombination<T>();
	newSymmetry->putSymmetry(l);
	newSymmetry->putSymmetry(this);
	return (Symmetry<T> *)newSymmetry;
}

