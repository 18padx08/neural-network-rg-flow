#include "FFT.h"
#include <assert.h>

using namespace std;
using namespace std::literals::complex_literals;

static double PI = 3.14159265359;

namespace fft {

	vector<complex<double>> fft(vector<double> data) {
		int n = data.size();
		if (n == 1)
		{
			return { data[0],0 };
		}
		assert(n % 2 == 0);
		vector<complex<double>> even(n / 2);
		vector<complex<double>> odd(n / 2);
		//#pragma omp parallel for
		for (int i = 0; i < n / 2; i++) {
			even[i] = data[2 * i];
			odd[i] = data[2 * i + 1];

		}
		vector<complex<double>> tmp_e(n / 2);
		tmp_e = fft(even);
		vector<complex<double>> tmp_o(n / 2);
		tmp_o = fft(odd);
		vector<complex<double>> full(n);
		//#pragma omp parallel for
		for (int i = 0; i < n / 2; i++) {
			full[i] = tmp_e[i] + exp(-2i * PI* (double)i / (double)n) *tmp_o[i];
			full[i + n / 2] = tmp_e[i] + exp(-2i * PI* (double)(i + n / 2) / (double)n)*tmp_o[i];
		}
		return full;
	}
	vector<complex<double>> fft(vector<complex<double>> data)
	{
		int n = data.size();
		if (n == 1)
		{
			return data;
		}
		assert(n % 2 == 0);
		vector<complex<double>> even(n / 2);
		vector<complex<double>> odd(n / 2);
//#pragma omp parallel for
		for (int i = 0; i < n / 2; i++) {
			even[i] = data[2 * i];
			odd[i] = data[2 * i + 1];
			
		}
		vector<complex<double>> tmp_e(n / 2);
		tmp_e = fft(even);
		vector<complex<double>> tmp_o(n / 2);
		tmp_o = fft(odd);
		vector<complex<double>> full(n);
//#pragma omp parallel for
		for (int i = 0; i < n/2; i++) {
			full[i] = tmp_e[i] + exp(-2i * PI* (double)i / (double)n) *tmp_o[i];
			full[i + n/2] = tmp_e[i] + exp(-2i * PI* (double)(i+n/2) / (double)n)*tmp_o[i];
		}
		return full;
	}
	vector<complex<double>> ifft(vector<complex<double>> data)
	{
		int n = data.size();
		if (n == 1)
		{
			return data;
		}
		assert(n % 2 == 0);
		vector<complex<double>> even(n / 2);
		vector<complex<double>> odd(n / 2);
		//#pragma omp parallel for
		for (int i = 0; i < n / 2; i++) {
			even[i] = data[2 * i];
			odd[i] = data[2 * i + 1];

		}
		vector<complex<double>> tmp_e(n / 2);
		tmp_e = ifft(even);
		vector<complex<double>> tmp_o(n / 2);
		tmp_o = ifft(odd);
		vector<complex<double>> full(n);
		//#pragma omp parallel for
		for (int i = 0; i < n / 2; i++) {
			full[i] =   ( tmp_e[i] + exp(2i * PI* (double)i / (double)n) *tmp_o[i]);
			full[i + n / 2] = ( tmp_e[i] + exp(2i * PI* (double)(i + n / 2) / (double)n)*tmp_o[i]);
		}
		return full;
	}
	vector<complex<double>> ifft(vector<double> data) {
		int n = data.size();
		if (n == 1)
		{
			return { data[0], 0 };
		}
		assert(n % 2 == 0);
		vector<complex<double>> even(n / 2);
		vector<complex<double>> odd(n / 2);
		//#pragma omp parallel for
		for (int i = 0; i < n / 2; i++) {
			even[i] = data[2 * i];
			odd[i] = data[2 * i + 1];

		}
		vector<complex<double>> tmp_e(n / 2);
		tmp_e = ifft(even);
		vector<complex<double>> tmp_o(n / 2);
		tmp_o = ifft(odd);
		vector<complex<double>> full(n);
		//#pragma omp parallel for
		for (int i = 0; i < n / 2; i++) {
			full[i] = (tmp_e[i] + exp(2i * PI* (double)i / (double)n) *tmp_o[i]);
			full[i + n / 2] = (tmp_e[i] + exp(2i * PI* (double)(i + n / 2) / (double)n)*tmp_o[i]);
		}
		return full;
	}
}
