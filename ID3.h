/*
 * ID3.h
 *
 *  Created on: Sep 19, 2015
 *      Author: th
 */

#ifndef ID3_H_
#define ID3_H_

#include <string>
#include <map>
#include <vector>

using std::vector;
using std::map;
using std::string;

class ID3{
private:
	bool has_real;
protected:
	int minLeaf; //segments为实数预处理时分为几段；minLeaf为叶子最小实例数
	string relation;
	string forecasting_attribute_name; //记录需要预测的属性名
	int forecasting_attribute_value_length; //记录需要预测的属性的分支数
	map<string, vector<string> > Attribute_Name_Values; //记录属性名和对应的属性值
	vector<string> Attribute_Names; //记录属性名
	vector<map<string, string> > Samples; //用来训练的实例
	vector<map<string, string> > Test_Samples; //用来测试的实例
	struct DecisionTreeNode{
		string attribute_name;
		string attribute_pre_value;
		string majority_forecasting_attribute_value; //当前剩余实例中最多的forecasting属性值
		int E; int N; //用于悲观剪枝E错误实例数；N当前总实例数
		vector<DecisionTreeNode*> children;
		DecisionTreeNode() :attribute_name(""), attribute_pre_value(""), majority_forecasting_attribute_value(""), E(-1), N(-1){};
	} *root;

	//熵相关函数
	//计算info([1,2,3,...])，number记录实例个数
	double Information(const map<string, int>& amount, int number)const;
	//求属性名下分支的信息值；当前考虑的实例集，当前考虑的属性名，属性名的属性值分支
	double ComputeEntropy(const vector<map<string, string> >& remain_Samples, string attribute_name, string attribute_value, int& number)const;
	//求当前实例集的信息值，并更新E和N
	double ComputeEntropy(const vector<map<string, string> >& remain_Samples, int& number)const;
	//属性名对应的信息增益
	double ComputeGain(const vector<map<string, string> >& remain_Samples, string attribute_name)const;

	//生成决策树的终止条件函数
	//判断remain中forecasting属性值是否都相同，若否返回"ImpurE"
	string IsPure(const vector<map<string, string> >& remain_Samples)const;
	//返回remain中最多的forecasting属性值
	string MajorityDecision(const vector<map<string, string> >& remain_Samples)const;
	//先序递归构建决策树
	virtual DecisionTreeNode* BuildDiscreteDecisionTree(DecisionTreeNode* current_node, vector<map<string, string> > remain_Samples, vector<string> remain_Attribute_Names);
	//先序递归打印决策树
	void PrintDecisionTree(DecisionTreeNode* node, int depth)const;
	//后序递归释放决策树
	void FreeDecisionTree(DecisionTreeNode* node);
	//返回has_real
	bool Gethasreal()const;
	void Sethasreal(bool flag);
public:
	//由于arff并不注明预测哪个属性，不妨默认为最后一个属性
	ID3(string path, int minLeaf = 2);
	void PrintDecisionTree()const;
	virtual void BuildDecisionTree();
	//按照百分比将Samples分为训练和测试的集合
	virtual void BuildDecisionTree(double percentage);
	//根据给定的单个样例进行预测
	virtual string Predict(map<string,string> example)const;
	//根据percentage划分的测试集进行测试，返回准确率
	double Predict()const;
	virtual void PessimisticPruning(const double z = 0.69){};
	void test();
	virtual ~ID3();
};

#endif
