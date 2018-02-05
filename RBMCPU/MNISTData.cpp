#include "MNISTData.h"
#include <iostream>
#include <fstream>
#include <assert.h>
#include <random>
#include <time.h>
using namespace std;

int32_t swapBytes(char *bytes) {
	int32_t numSamples = 0;
	numSamples |= (*bytes & 0xFF) << 24;
	numSamples |= (*(bytes + 1) & 0xFF) << 16;
	numSamples |= (*(bytes + 2) & 0xFF) << 8;
	numSamples |= (*(bytes + 3) & 0xFF);
	return numSamples;
}

MNISTData::MNISTData()
{
	
	//openfiles
	std::ifstream images("../data/train-images.idx3-ubyte", ios::in | ios::binary | ios::ate);
	std::ifstream labels("../data/train-labels.idx1-ubyte", ios::in | ios::binary | ios::ate);
	images.seekg(0, ios::beg);
	labels.seekg(0, ios::beg);
	char *magicNumber = (char*)malloc(sizeof(int32_t));
	char *num_samples = (char *)malloc(sizeof(int32_t));
	//read magicnumber and sample size of labels
	labels.read(magicNumber, sizeof(int32_t));
	labels.read(num_samples, sizeof(int32_t));

	//read magicnumber size and number of samples from images
	char *sampleSize = new char[sizeof(int32_t)];
	char *rows = new char[sizeof(int32_t)];
	char *columns = new char[sizeof(int32_t)];
	images.read(magicNumber, sizeof(int32_t));
	images.read(sampleSize, sizeof(int32_t));
	images.read(rows, sizeof(int32_t));
	images.read(columns, sizeof(int32_t));

	int32_t ss = swapBytes(sampleSize);
	int32_t r = swapBytes(rows);
	int32_t co = swapBytes(columns);
	//convert endianess
	int32_t magic = swapBytes(magicNumber);
	int32_t numsamp = swapBytes(num_samples);
	assert(numsamp == ss);
	//init arrays
	this->labels = (char *)malloc(numsamp);
	this->images = (char **)malloc(ss*sizeof(char*));
	for (int b = 0; b < ss; b++) {
		this->images[b] = (char*)malloc(co*r*sizeof(char));
	}
	//set width and height of pictures
	this->height = r;
	this->width = co;
	this->samples = numsamp;
	//read all lables into array
	labels.read(this->labels, numsamp);

	//read images into array
	for(int i =0; i < numsamp; i++) {
		images.read(this->images[i], co*r);
		
	}

	images.close();
	labels.close();
	delete(magicNumber);
	delete(num_samples);
	delete(sampleSize);
	delete(rows);
	delete(columns);
}

double ** MNISTData::getBatch(int batchSize)
{
	double **sampleBatch = (double**) malloc(batchSize * sizeof(double *));
	for (int i = 0; i < batchSize; i++) {
		sampleBatch[i] =  (double*)malloc(sizeof(double)*this->width * this->height);
		//only take the ones which are a 3 or 6
		int random = rand() % this->samples;
		unsigned char c = this->labels[random];
	
			random = rand() % this->samples;
			c = this->labels[random];
		
		int offset = 0;
		for (int offset = 0; offset < this->width*this->height; offset ++) {
			unsigned char c = this->images[random][offset];
			sampleBatch[i][offset] = std::round((int)c/255.0); //normalize to 0 and 1
		}
	}
	std::cout << sampleBatch[0][1];
	return sampleBatch;
}
