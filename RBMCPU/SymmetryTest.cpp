#include "SymmetryTest.h"
#include "SymmetryCombination.cpp"
#include "TranslationSymmetry.cpp"
#include <initializer_list>
#include <iostream>
#include <stdio.h>
#include <string>

SymmetryTest::SymmetryTest()
{
}


SymmetryTest::~SymmetryTest()
{
}

void SymmetryTest::runSymmetryTest()
{
	int length = 1000;
	int *input = new int[1000];
	int *output = new int[1000];
	for (int i = 0; i < length; i++) {
		input[i] = i % 2 == 0 ? i : -i;
		output[i] = 0;
	}
	
	TranslationSymmetry<int> *ts = new TranslationSymmetry<int>();
	TranslationSymmetry<int> *ts2 = new TranslationSymmetry<int>();
	Symmetry<int> *ts3 = (*ts) * ts2;
	(*ts3)(input, output, length);
	for (int i = 0; i < length-2; i++) {
		std::cout << output[i] << " == " << input[i+2] << " => "  << (output[i] == input[i+2] ? "[" + std::to_string(i) +"]" + "PASSED" : "[" + std::to_string(i) + "]" + "NOT PASSED") << std::endl;
	}
	std::cout << output[length - 1] << " == " << input[1] << " => " << (output[length-1] == input[1] ? "[" + std::to_string(length-1) + "]" + "PASSED" : "[" + std::to_string(length -1) + "]" + "NOT PASSED") << std::endl;
	std::cout << output[length - 2] << " == " << input[0] << " => " << (output[length - 2] == input[0] ? "[" + std::to_string(length - 2) + "]" + "PASSED" : "[" + std::to_string(length - 2) + "]" + "NOT PASSED") << std::endl;

}
