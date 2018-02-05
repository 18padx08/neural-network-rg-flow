#include "SymmetryCombination.h"
template<class T>
 int SymmetryCombination<T>::operator()(T input[], T output[], size_t length)
{
	 for (int i = 0; i < std::count(this->symList.begin(), this->symList.end()); i++) {
		 symList[i](input, output, length);
	 }
	 return 0;
}
/*
template<class T>
 Symmetry<T> SymmetryCombination<T>::operator*(Symmetry<T> l)
{
	 this->symList.push_back(l);
	 return this;
}
*/
