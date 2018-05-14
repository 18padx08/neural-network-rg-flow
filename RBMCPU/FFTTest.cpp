#include "FFTTest.h"
#include "FFT.cpp"
#include <vector>
#include <complex>
#include <iostream>
#include <chrono>
using namespace std;
typedef complex<double> dcomplex;

FFTTest::FFTTest()
{
}


FFTTest::~FFTTest()
{
}


void FFTTest::run()
{
	int count = std::pow(2, 12);
	vector<dcomplex> data(count);
	for (int i = 0; i < count; i++) {
		data[i] = std::sin(i*2*PI/count);
	}
	std::cout << std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	auto tmp = fft::fft(data);
	auto elapsed = std::chrono::high_resolution_clock::now() - start;

	std::cout << std::endl;
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << duration << std::endl;
}
