#pragma once
class Phi4Test
{
public:
	Phi4Test();
	~Phi4Test();
	void run();
	void runCorrTest();
	void runNetworkTest();
	void runMassTest(double kappa = 0.35, int chainsize = 4096);
};

