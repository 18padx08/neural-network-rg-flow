#include "TestFFTUpdate.h"
#include "Phi1D.h"
#include <iostream>


TestFFTUpdate::TestFFTUpdate()
{
}


TestFFTUpdate::~TestFFTUpdate()
{
}

void TestFFTUpdate::run()
{
	Phi1D cluster(64,0.2,0,0,0);
	Phi1D fft(64, 0.2, 0, 0, 0);
	double diff = 0;
	int trials = 10000;
	cluster.thermalize();
	//fft.thermalize();
	fft.fftUpdate();
	for (int i = 0; i < trials; i++) {
		auto clTmp = cluster.volumeAverage();
		auto fftTmp = fft.volumeAverage();
		diff += clTmp - fftTmp ;
		cluster.monteCarloSweep();
		fft.fftUpdate();
		//std::cout << "Current difference " << clTmp - fftTmp;
	}
	std::cout << "Total difference: " << diff / trials << std::endl;
}
