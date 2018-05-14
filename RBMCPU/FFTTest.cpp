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
	int count = std::pow(2, 4);
	vector<dcomplex> data(count);
	for (int i = 0; i < count; i++) {
		data[i] = std::sin(i*2*PI/count);
	}
	std::cout << std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	auto tmp = fft::fft(data);
	auto elapsed = std::chrono::high_resolution_clock::now() - start;
	auto reverse = fft::ifft(tmp);
	for (int i = 0; i < count; i++) {
		reverse[i] = 1.0 / count * reverse[i];
		if (abs(reverse[i].real() - data[i].real()) > 1e-6 || abs(reverse[i].imag() - data[i].imag()) > 1e-6) {
			//std::cout << "Wrong ifft  " << abs( reverse[i].real() - data[i].real()) << "   " << abs( reverse[i].imag() - data[i].imag()) << std::endl;
			std::cout << reverse[i] << "  " << data[i] << std::endl;
		}
	}
	std::cout << std::endl;
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	std::cout << duration << std::endl;
}

void FFTTest::runSymmetryTest()
{
	// -1.0000 -0.8824 -0.7647 -0.6471 -0.5294 -0.4118 -0.2941 -0.1765 -0.0588 0.0588 0.1765 0.2941 0.4118 0.5294 0.6471 0.7647 0.8824
	vector<dcomplex> data = { {0,0}, {3,0}, {2,0}, {3,0},{ -1,0 },{ 3,0 },{ 2,0 },{ 3,0 } };
	vector<dcomplex> fftData = { {1,0},{-3,0},{-2,0},{4,0} };
	auto output = fft::ifft(data);
	for (auto &&a : output) {
		std::cout << a << std::endl;
	}

}
