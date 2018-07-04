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
	this->dimensions.clear();
}

template<class T>
T& LatticeObject<T>::operator[](const vector<int> index)
{
	//flatten dimensions
	//periodic boundary conditions
	int dim = index[0] <0 ? abs(dimensions[0] - index[0]) % dimensions[0] : index[0] % dimensions[0];
	int lastDimensions = dimensions[0];
	for (int i = 1; i < index.size(); i++) {
		if (i > dimensions.size() - 1)
		{
			break;
		}
		dim += (index[i] < 0 ? abs(dimensions[i] - index[i]) % dimensions[i] : index[i] % dimensions[i]) * lastDimensions;
		lastDimensions *= dimensions[i];
	}
	return elements[dim];
}
