#pragma once
#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>
#include <stdexcept>

using namespace std;

template <class T>
class LatticeObject
{
public:
	LatticeObject<T>();
	LatticeObject<T>(const vector<int> size);
	~LatticeObject<T>();
	T& operator[] (const vector<int> index);
	vector<int> dimensions;
	T *lattice;
	int latticeSize;

};


