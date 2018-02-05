#pragma once
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
};
