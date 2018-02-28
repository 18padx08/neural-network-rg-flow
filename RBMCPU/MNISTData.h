#pragma once
#include <vector>
class MNISTData {
private:
	int width;
	int height;
	int samples;
	char **images;
	char *labels;
public:
	MNISTData();
	double **getBatch(int batchSize);
	std::vector<std::vector<double>> getVectorizedBatch(int batchSize);
};
