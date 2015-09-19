/*
 * main.cpp
 *
 *  Created on: Sep 19, 2015
 *      Author: th
 */

#include "ID3.h"
#include "C4d5.h"
#include <iostream>

using std::cout;

int main(int argc, char* argv[])
{
	/**********test************/
	//map<string, string> example;
	//example.insert(map<string, string>::value_type("outlook", "rainy"));
	//example.insert(map<string, string>::value_type("temperature", "cool"));
	//example.insert(map<string, string>::value_type("humidity", "normal"));
	//example.insert(map<string, string>::value_type("windy", "strong"));
	//example.insert(map<string, string>::value_type("play", "yes"));
	//C4d5 *test = new C4d5("test.txt");
	//test->BuildDecisionTree();
	//test->PessimisticPruning();
	//test->PrintDecisionTree();
	//cout << test->Predict(example);
	//test->~C4d5();

	/*****Iris_test****/
	/*ID3 *test = new ID3("iris_test.txt");
	test->BuildDecisionTree(0.15);
	test->PrintDecisionTree();
	cout << "准确率：" << test->Predict() * 100 << "%";
	test->~ID3();*/
	/******************/

	/*****Iris_test2****/
	/*C4d5 *test = new C4d5("iris_test.txt");
	test->BuildDecisionTree(0.15);
	test->PessimisticPruning();
	test->PrintDecisionTree();
	cout << "准确率：" << test->Predict() * 100 << "%";
	test->~C4d5();*/
	/******************/

	/*****Iris_test2****/
	//map<string, string> example;
	//example.insert(map<string, string>::value_type("sepallength", "5.0"));
	//example.insert(map<string, string>::value_type("sepalwidth", "2.3"));
	//example.insert(map<string, string>::value_type("petallength", "3.3"));
	//example.insert(map<string, string>::value_type("petalwidth", "1.0"));
	//C4d5 *test = new C4d5("iris_test.txt");
	//test->BuildDecisionTree();
	//test->PessimisticPruning();
	//test->PrintDecisionTree();
	//cout << test->Predict(example);
	//test->~C4d5();
	/******************/

	/*****Iris_mix****/
	C4d5 *test = new C4d5("test_mix.txt");
	test->BuildDecisionTree(0.1);
	test->PessimisticPruning();
	test->PrintDecisionTree();
	cout << "准确率：" << test->Predict() * 100 << "%";
	test->~C4d5();
	/******************/

	/**************/
	//C4d5 *test = new C4d5("iris_test.txt");
	//test->test();
	return 0;
}



