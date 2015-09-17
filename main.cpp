/*
 * main.cpp
 *
 *  Created on: Mar 19, 2015
 *      Author: th
 */
#include "ID3.h"
#include "C4d5.h"
#include <iostream>
#include <stdio.h>

using std::cout;

int main(void)
{
	//map<string, string> example;
	//example.insert(map<string, string>::value_type("outlook", "rainy"));
	//example.insert(map<string, string>::value_type("temperature", "cool"));
	//example.insert(map<string, string>::value_type("humidity", "normal"));
	//example.insert(map<string, string>::value_type("windy", "strong"));
	//example.insert(map<string, string>::value_type("play", "yes"));
	ID3 *test = new C4d5("iris_test.txt");
//	ID3 *test = new ID3("iris_test.txt");
//	ID3 *test = new ID3("test.txt");
	test->BuildDecisionTree(0.15);
//	test->BuildDecisionTree();
	test->PessimisticPruning();
	test->PrintDecisionTree();
//	cout << test->Predict(example);
	cout << test->Predict();
	test->~ID3();
	return 0;
}



