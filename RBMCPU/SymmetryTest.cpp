#include "SymmetryTest.h"
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
	
	TranslationSymmetry<int> ts;
	ts(input, output, length);
	for (int i = 0; i < length-1; i++) {
		std::cout << output[i] << " == " << input[i+1] << " => "  << (output[i] == input[i+1] ? "[" + std::to_string(i) +"]" + "PASSED" : "[" + std::to_string(i) + "]" + "NOT PASSED") << std::endl;
	}
	std::cout << output[length - 1] << " == " << input[0] << " => " << (output[length-1] == input[0] ? "[" + std::to_string(length-1) + "]" + "PASSED" : "[" + std::to_string(length -1) + "]" + "NOT PASSED") << std::endl;

}
