#include "TestFFTUpdate.h"
#include "Phi1D.h"
#include <iostream>
#include <chrono>
#include "FFT.cpp"
using namespace std::chrono_literals;
TestFFTUpdate::TestFFTUpdate()
{
}


TestFFTUpdate::~TestFFTUpdate()
{
}

void TestFFTUpdate::run()
{
	Phi1D cluster(256,0.4,0,0,0);
	Phi1D fft(256, 0.4, 0, 0, 0);
	double diff = 0;
	int trials = 10000;
	cluster.thermalize();
	//fft.thermalize();
	auto clusterTime = 0ns;
	auto fftTime = 0ns;
	fft.fftUpdate();
	double totalfft = 0;
	double totalcluster = 0;
	for (int i = 0; i < trials; i++) {
		auto clTmp = cluster.getCorrelationLength();
		auto fftTmp = fft.getCorrelationLength();
		auto test = cluster.getConfiguration();
		auto fftofit = fft::fft(test);
		diff += clTmp - fftTmp ;
		totalfft += fftTmp;
		totalcluster += clTmp;
		auto start = std::chrono::high_resolution_clock::now();
		cluster.monteCarloSweep();
		clusterTime += std::chrono::high_resolution_clock::now() - start;
		start = std::chrono::high_resolution_clock::now();
		fft.fftUpdate();
		fftTime += std::chrono::high_resolution_clock::now() - start;
		//std::cout << "Current difference " << clTmp << "  " << fftTmp <<std::endl;
	}
	std::cout << "Total difference: " << diff / trials << std::endl;
	std::cout << "Cluster Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(clusterTime).count() << " vs fft Time " << std::chrono::duration_cast<std::chrono::milliseconds>(fftTime).count() << std::endl;
	std::cout << "Current difference " << totalcluster / trials << "  " << totalfft/trials << std::endl;
}

void TestFFTUpdate::runFFTCompareToNewHidden()
{
	double kappa = 0.45;
	Phi1D fftUpdater(512, kappa, 0, 0, 0);
	//_sleep(1000);
	Phi1D fftDec(512, kappa*kappa/(1.0-2.0*kappa*kappa), 0, 0, 0);
	int trials = 5000;
	double diff = 0;
	double totalCl = 0;
	double totalDec = 0;
	for (int i = 0; i < trials; i++) {
		fftUpdater.fftUpdate();
		fftDec.fftUpdate();

		fftDec.rescaleFields(1.0/std::sqrt(1 - 2 * kappa*kappa));
		auto tmpCl = fftUpdater.getCorrelationLength(2);
		auto tmpDec = fftDec.getCorrelationLength();
		diff += tmpCl - tmpDec ;
		totalCl += tmpCl;
		totalDec += tmpDec;
	}
	std::cout << "Difference in correlation length: " << diff / trials << std::endl;
	//std::cout << "Corr length " << fftUpdater.getCorrelationLength(2) << "  " << fftDec.getCorrelationLength() << std::endl;
	std::cout << "Correlation Length: " << totalCl / trials << "  " << totalDec /trials << std::endl;
}

