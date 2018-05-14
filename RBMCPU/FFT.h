#pragma once
#include <vector>
#include <complex>
using namespace std;
namespace fft {
		static vector<complex<double>> fft(vector<complex<double>> data);	
		static vector<complex<double>> fft(vector<double> data);
		static vector<complex<double>> ifft(vector<complex<double>> data);
		static vector<complex<double>> ifft(vector<double> data);
}

