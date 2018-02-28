#include "SymmetryCombination.h"
#include <iostream>
#include <stdio.h>
template<class T>
 int SymmetryCombination<T>::operator()(T input[], T output[], size_t length)
{
	 auto tmpInput = new T[length];
	 auto tmpOutput = new T[length];
	 for (int i = 0; i < this->symList.size(); i++) {
		 std::cout << i << std::endl;
		 if (i == 0) {
			 (*symList[i])(input, tmpInput, length);
		 }
		 else if (i == this->symList.size() -1) {
			 (*symList[i])(tmpInput, output, length);
		 }
		 else {
			 (*symList[i])(tmpInput, tmpOutput, length);
			 tmpInput = tmpOutput;
		 }
	 }
	 delete[](tmpInput);
	 delete[](tmpOutput);
	 return 0;
}

 template<class T>
 vector<T> SymmetryCombination<T>::operator()(vector<T>& input)
 {
	 return vector<T>();
 }

 template<class T>
 void SymmetryCombination<T>::putSymmetry(Symmetry<T>* sym)
 {
	 this->symList.push_back(sym);
 }

template<class T>
 Symmetry<T> *SymmetryCombination<T>::operator*(Symmetry<T> *l)
{
	 this->symList.push_back(l);
	 return this;
}

