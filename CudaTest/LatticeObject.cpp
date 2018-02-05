#include "LatticeObject.h"
#include <assert.h>

template<class T>
inline LatticeObject<T>::LatticeObject(const vector<int> size) : dimensions()
{
	this->latticeSize = 1;
	for (auto i : size) {
		latticeSize *= i;
		this->dimensions.push_back(i);
	}
	this->lattice = (T*)malloc(latticeSize*sizeof(T));
	for (int i = 0; i < latticeSize; i++) {
		this->lattice[i] = T();
	}
}

template<class T>
LatticeObject<T>::~LatticeObject()
{
	if (this->lattice != nullptr) {
		free(this->lattice);
	}
}

template<class T>
T& LatticeObject<T>::operator[](const vector<int> index)
{
	//flatten dimensions
	//periodic boundary conditions
	assert(index.size() == this->dimensions.size());
	int flatIndex = index[0];
	int lastDimension = 1;
	for (int i = 1; i < index.size(); i++) {
		int tmpIndex = index[i];
		if (tmpIndex >= this->dimensions[i]) {
			//index is out of bounds so start over at zero
			tmpIndex = tmpIndex % this->dimensions[i];
		}
		else if(tmpIndex < 0) {
			//index is negative start at last element ensure to be inside bounds
			tmpIndex = tmpIndex % this->dimensions[i];
			tmpIndex = this->dimensions[i] + tmpIndex;
		}
		flatIndex += tmpIndex * this->dimensions[i-1] * lastDimension;
		lastDimension *= this->dimensions[i];
	}
	if (this->dimensions.size() == 1) {
		if (flatIndex >= this->dimensions[0]) {
			flatIndex = flatIndex % this->dimensions[0];
		}
		if (flatIndex < 0) {
			//we have the one dimensional special case
			flatIndex = flatIndex % this->dimensions[0];
			flatIndex = this->dimensions[0] + flatIndex;
		}
	}
	
	return this->lattice[flatIndex];
}
