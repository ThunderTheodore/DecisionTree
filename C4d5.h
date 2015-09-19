/*
 * C4d5.h
 *
 *  Created on: Sep 19, 2015
 *      Author: th
 */

#ifndef C4d5_H_
#define C4d5_H_

#include "ID3.h"

using std::multimap;

class C4d5 :public ID3{
protected:
	//C4.5求分裂信息度量
	double ComputeSplitEntropy(const vector<map<string, string> >& remain_Samples, string attribute_name, int& number)const;
	//C4.5求信息增益
	double ComputeGainRatio(const vector<map<string, string> >& remain_Samples, string attribute_name)const;

	//先序递归构建决策树
	DecisionTreeNode* BuildDiscreteDecisionTree(DecisionTreeNode* current_node, vector<map<string, string> > remain_Samples, vector<string> remain_Attribute_Names);
	//记录结点中的E和N和majority_forecasting_attribute_value
	void RecordEN(const vector<map<string, string> >& remain_Samples, DecisionTreeNode* current_node);
	//计算悲观的误差率，若N为0返回0
	double ErrorRate(const int E, const int N, const double z)const;
	//子树置换，若发生替换返回1
	bool SubtreeReplacement(DecisionTreeNode* current_node, const double z);
	//剪枝，用当前节点下的叶子代替当前节点
	void Pruning(DecisionTreeNode* current_node, const double z);
	//先序遍历树寻找需要剪枝的节点
	bool PessimisticPruning(DecisionTreeNode* current_node, const double z);
	//根据置信度c求标准正态分布单侧z值
	double ComputeZ(const double p)const;

	//先序递归构建带连续属性的决策树
	DecisionTreeNode* BuildContinuousDecisionTree(DecisionTreeNode* current_node, vector<map<string, string> > remain_Samples, vector<string> remain_Attribute_Names, map<string, multimap<double, string> > remain_Numerical_Attribute_Samples);
	//根据remain_Samples生成remain_Numerical_Attribute_Samples
	void PreProcessorForContinuous(const vector<map<string, string> >& remain_Samples, map<string, multimap<double, string> >& remain_Numerical_Attribute_Samples)const;
	//根据连续数值和分割点位置求信息熵
	double ComputeContinuousEntropy(const multimap<double, string>& remain_Numerical_Samples, const multimap<double, string>::const_iterator& index)const;
	//根据连续数值和分割点位置求信息增益率
	double ComputeContinuousGainRatio(const multimap<double, string>& remain_Numerical_Samples, const multimap<double, string>::const_iterator& index)const;
	//找到所有分割点，返回最佳分割点的信息增益率，index为最佳分割点的位置
	double FindDividingPoint(const multimap<double, string>& remain_Numerical_Samples, multimap<double, string>::const_iterator& index)const;
	//离散化，返回<[index]和>=[index]两部分的string
	vector<string> DivideContinuousIntoTwoParts(const multimap<double, string>::const_iterator& index)const;
	//生成子节点的remain_Numercial_Attribute_Samples和remain_Samples；若new_remain_Samples的size没变返回1，size为0返回2，正常返回0
	int UpdateContinuousSamples(const vector<map<string, string> >& remain_Samples, vector<map<string, string> >& new_remain_Samples, map<string, multimap<double, string> >& new_remain_Numerical_Attribute_Samples, string attribute_name, string attribute_value);
	//判断continuous_value是否满足不等式inequality
	bool IsInequalitySatisfied(const string& inequality, const string& continuous_value)const;

	//将连续属性值等分为segments份
	void PreProcessor(int segments);
public:
	C4d5(string path, int minLeaf = 2) :ID3(path, minLeaf){};
	~C4d5(){};
	//剪枝函数
	void PessimisticPruning(const double p = 0.25);
	//flag控制是否预先离散化
	void BuildDecisionTree(bool flag = 0, int segments = 6);
	void BuildDecisionTree(double percentage, bool flag = 0, int segments = 6);

	string Predict(map<string, string> example)const;
	double Predict()const;
	void test();
};
#endif
