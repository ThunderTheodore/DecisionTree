/*
 * C4d5.cpp
 *
 *  Created on: Sep 19, 2015
 *      Author: th
 */

#include "C4d5.h"
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
using std::cout;
using std::stringstream;

double C4d5::ComputeGainRatio(const vector<map<string, string> >& remain_Samples, string attribute_name)const{
	int number;
	double temp = ComputeSplitEntropy(remain_Samples, attribute_name, number);
	if (!temp) return 65535; //分裂信息值为0时返回最大值
	else return ComputeGain(remain_Samples, attribute_name) / temp;
}

ID3::DecisionTreeNode* C4d5::BuildDiscreteDecisionTree(DecisionTreeNode* current_node, vector<map<string, string> > remain_Samples, vector<string> remain_Attribute_Names){
	//计算E和N
	RecordEN(remain_Samples, current_node);
	//终止条件
	if (remain_Samples.size() <= (unsigned)minLeaf){ //剩余实例数量小于阈值
		current_node->attribute_name = current_node->majority_forecasting_attribute_value;
		return current_node;
	}
	if (!current_node->E){ //信息值为0
		current_node->attribute_name = current_node->majority_forecasting_attribute_value;
		return current_node;
	}
	if (remain_Attribute_Names.empty()){ //没有剩余的属性名
		current_node->attribute_name = current_node->majority_forecasting_attribute_value;
		return current_node;
	}
	//找最大信息增益
	double max_gainratio = -1, temp_gainratio;
	vector<string>::iterator chosen_attribute_name;
	vector<string>::iterator i;
	for (i = remain_Attribute_Names.begin(); i != remain_Attribute_Names.end(); ++i){
		temp_gainratio = ComputeGainRatio(remain_Samples, *i);
		if (temp_gainratio > max_gainratio){
			max_gainratio = temp_gainratio;
			chosen_attribute_name = i;
		}
	}
	current_node->attribute_name = *chosen_attribute_name;
	//构建子节点的remain_Attribute_Names
	vector<map<string, string> > new_remain_Samples;
	vector<string> new_remain_Attribute_Names;
	for (i = remain_Attribute_Names.begin(); i != remain_Attribute_Names.end(); ++i){
		if ((*i).compare(*chosen_attribute_name))
			new_remain_Attribute_Names.push_back(*i);
	}
	//根据确定的name生成相应value的子节点
	for (i = Attribute_Name_Values[*chosen_attribute_name].begin(); i != Attribute_Name_Values[*chosen_attribute_name].end(); ++i){
		DecisionTreeNode* child_node = new DecisionTreeNode();
		child_node->attribute_pre_value = *i;
		for (vector<map<string, string> >::iterator j = remain_Samples.begin(); j != remain_Samples.end(); ++j){ //构建子节点的remain_Samples
			if (!(*i).compare(j->at(*chosen_attribute_name))){
				new_remain_Samples.push_back(*j);
			}
		}
		//递归生成子节点
		BuildDiscreteDecisionTree(child_node, new_remain_Samples, new_remain_Attribute_Names);
		current_node->children.push_back(child_node);
		new_remain_Samples.clear();
	}
	return current_node;
}

void C4d5::PreProcessorForContinuous(const vector<map<string, string> >& remain_Samples, map<string, multimap<double, string> >& remain_Numerical_Attribute_Samples)const{
	double num_temp;
	stringstream ss_temp;
	for (map<string, vector<string> >::const_iterator i = Attribute_Name_Values.begin(); i != Attribute_Name_Values.end(); ++i){
		if (!(i->second[0].compare("ReaL"))){ //找到数值属性名
			multimap<double, string> numerical_samples;
			for (vector<map<string, string> >::const_iterator j = remain_Samples.begin(); j != remain_Samples.end(); ++j){
				ss_temp << j->at(i->first);
				ss_temp >> num_temp;
				ss_temp.clear();
				numerical_samples.insert(multimap<double, string>::value_type(num_temp, j->at(forecasting_attribute_name)));
			}
			remain_Numerical_Attribute_Samples.insert(map<string, multimap<double, string> >::value_type(i->first, numerical_samples));
		}
	}
}

double C4d5::ComputeContinuousEntropy(const multimap<double, string>& remain_Numerical_Samples, const multimap<double, string>::const_iterator& index)const{
	int number1 = 0, number2 = 0;
	double info1 = 0, info2 = 0;
	map<string, int> amount;
	multimap<double, string>::const_iterator i;
	for (i = remain_Numerical_Samples.begin(); i != index; ++i){
		++amount[i->second];
		++number1;
	}
	info1 = Information(amount, number1);
	amount.clear();
	for (; i != remain_Numerical_Samples.end(); ++i){
		++amount[i->second];
		++number2;
	}
	info2 = Information(amount, number2);
	return (double)number1 / (number1 + number2)*info1 + (double)number2 / (number1 + number2)*info2;
}

double C4d5::ComputeContinuousGainRatio(const multimap<double, string>& remain_Numerical_Samples, const multimap<double, string>::const_iterator& index)const{
	int number1 = 0, number2 = 0, number_all;
	double info1 = 0, info2 = 0, info_all, info_num, info_split;
	map<string, int> amount, amount_all;
	multimap<double, string>::const_iterator i;
	for (i = remain_Numerical_Samples.begin(); i != index; ++i){
		++amount[i->second];
		++amount_all[i->second];
		++number1;
	}
	info1 = Information(amount, number1);
	amount.clear();
	for (; i != remain_Numerical_Samples.end(); ++i){
		++amount[i->second];
		++amount_all[i->second];
		++number2;
	}
	info2 = Information(amount, number2);
	number_all = number1 + number2;
	info_all = Information(amount_all, number_all);
	info_num = (double)number1 / number_all*info1 + (double)number2 / number_all*info2;
	info_split = -(double)number1 / number_all*log((double)number1 / number_all) / log(2.0) - (double)number2 / number_all*log((double)number2 / number_all) / log(2.0);
	return (info_all - info_num) / info_split;
}

double C4d5::FindDividingPoint(const multimap<double, string>& remain_Numerical_Samples, multimap<double, string>::const_iterator& index)const{
	multimap<double, string>::const_iterator i = remain_Numerical_Samples.begin();
	string forecasting_value_temp = i->second;
	double temp;
	double min = 9999999; //记录最小信息熵(最大信息增益)
	multimap<double, string>::const_iterator min_index; //记录相应分割点
	for (++i; i != remain_Numerical_Samples.end(); ++i){
		if (i->second.compare(forecasting_value_temp)){ //类别不同
			temp = ComputeContinuousEntropy(remain_Numerical_Samples, i);
			forecasting_value_temp = i->second;
			if (temp < min){
				min = temp;
				min_index = i;
			}
		}
	}
	index = min_index;
	return ComputeContinuousGainRatio(remain_Numerical_Samples,min_index);
}

vector<string> C4d5::DivideContinuousIntoTwoParts(const multimap<double, string>::const_iterator& index)const{
	string split_value;
	stringstream ss_temp;
	vector<string> lr_attribute_value;
	ss_temp << index->first;
	ss_temp >> split_value;
	lr_attribute_value.push_back("<" + split_value);
	lr_attribute_value.push_back(">=" + split_value);
	return lr_attribute_value;
}

bool C4d5::IsInequalitySatisfied(const string& inequality, const string& continuous_value)const{
	double data, split_value;
	stringstream ss_temp;
	ss_temp.clear();
	ss_temp << continuous_value;
	ss_temp >> data;
	if (inequality[0] == '<'){
		ss_temp.clear();
		ss_temp << inequality.substr(1, -1);
		ss_temp >> split_value;
		if (data < split_value) return 1;
		else return 0;
	}
	else if (inequality[0] == '>'&&inequality[1] == '='){
		ss_temp.clear();
		ss_temp << inequality.substr(2, -1);
		ss_temp >> split_value;
		if (data >= split_value) return 1;
		else return 0;
	}
	return 0;
}

int C4d5::UpdateContinuousSamples(const vector<map<string, string> >& remain_Samples, vector<map<string, string> >& new_remain_Samples, map<string, multimap<double, string> >& new_remain_Numerical_Attribute_Samples, string attribute_name, string attribute_value){
	double split_value, data;
	stringstream ss_temp;
	if (attribute_value[0] == '<'){
		ss_temp << attribute_value.substr(1, -1);
		ss_temp >> split_value;
		ss_temp.clear();
		for (vector<map<string, string> >::const_iterator i = remain_Samples.begin(); i != remain_Samples.end(); ++i){
			ss_temp << i->at(attribute_name);
			ss_temp >> data;
			ss_temp.clear();
			if (data < split_value){
				new_remain_Samples.push_back(*i);
			}
		}
	}
	else if (attribute_value[0] == '>'&&attribute_value[1] == '='){
		ss_temp << attribute_value.substr(2, -1);
		ss_temp >> split_value;
		ss_temp.clear();
		for (vector<map<string, string> >::const_iterator i = remain_Samples.begin(); i != remain_Samples.end(); ++i){
			ss_temp << i->at(attribute_name);
			ss_temp >> data;
			ss_temp.clear();
			if (data >= split_value){
				new_remain_Samples.push_back(*i);
			}
		}
	}
	if (remain_Samples.size() == new_remain_Samples.size()) return 1;
	if (new_remain_Samples.empty()) return 2;
	PreProcessorForContinuous(new_remain_Samples, new_remain_Numerical_Attribute_Samples);
	return 0;
}

void C4d5::BuildDecisionTree(bool flag,int segments){
	vector<string> current_Attribute_Names;
	//不考虑forecasting的属性名
	current_Attribute_Names.insert(current_Attribute_Names.begin(), Attribute_Names.begin(), Attribute_Names.end() - 1);
	if (Gethasreal()){
		if (!flag){
			map<string, multimap<double, string> > Numerical_Attribute_Samples; //存放连续属性 属性名，multimap<数值，类别>
			PreProcessorForContinuous(Samples, Numerical_Attribute_Samples);
			BuildContinuousDecisionTree(root, Samples, current_Attribute_Names, Numerical_Attribute_Samples);
		}
		else{ //预离散化
			PreProcessor(segments);
			BuildDiscreteDecisionTree(root, Samples, current_Attribute_Names);
		}
	}
	else{
		BuildDiscreteDecisionTree(root, Samples, current_Attribute_Names);
	}
}

void C4d5::BuildDecisionTree(double percentage, bool flag, int segments){
	if (percentage < 0 || percentage>1){
		cout << "BuildDecisionTree(double percentage)参数错误";
		return;
	}
	if (flag){
		PreProcessor(segments);
	}
	vector<string> current_Attribute_Names;
	current_Attribute_Names.insert(current_Attribute_Names.begin(), Attribute_Names.begin(), Attribute_Names.end() - 1);
	int training_size = (int)((1.0 - percentage)*Samples.size());
	int test_size = Samples.size() - training_size;
	Test_Samples.insert(Test_Samples.begin(), Samples.end() - test_size, Samples.end()); //后percentage比例的实例作为测试
	Samples.erase(Samples.end() - test_size, Samples.end());
	if (Gethasreal()){
		if (!flag){
			map<string, multimap<double, string> > Numerical_Attribute_Samples; //存放连续属性 属性名，multimap<数值，类别>
			PreProcessorForContinuous(Samples, Numerical_Attribute_Samples);
			BuildContinuousDecisionTree(root, Samples, current_Attribute_Names, Numerical_Attribute_Samples);
		}
	}
	else{
		BuildDiscreteDecisionTree(root, Samples, current_Attribute_Names);
	}
}

ID3::DecisionTreeNode* C4d5::BuildContinuousDecisionTree(DecisionTreeNode* current_node, vector<map<string, string> > remain_Samples, vector<string> remain_Attribute_Names, map<string, multimap<double, string> > remain_Numerical_Attribute_Samples){
	//计算E和N
	RecordEN(remain_Samples, current_node);
	//终止条件
	if (remain_Samples.size() <= (unsigned)minLeaf){ //剩余实例数量小于阈值
		current_node->attribute_name = current_node->majority_forecasting_attribute_value;
		return current_node;
	}
	if (!current_node->E){ //信息值为0
		current_node->attribute_name = current_node->majority_forecasting_attribute_value;
		return current_node;
	}
	if (remain_Attribute_Names.empty()){ //没有剩余的属性名
		current_node->attribute_name = current_node->majority_forecasting_attribute_value;
		return current_node;
	}
	//找最大信息增益
	double max_gainratio = -1, temp_gainratio;
	vector<string>::iterator chosen_attribute_name;
	multimap<double, string>::const_iterator chosen_index;
	vector<string>::iterator i;
	for (i = remain_Attribute_Names.begin(); i != remain_Attribute_Names.end(); ++i){
		if (Attribute_Name_Values[*i][0].compare("ReaL")){ //离散
			temp_gainratio = ComputeGainRatio(remain_Samples, *i);
			if (temp_gainratio > max_gainratio){
				max_gainratio = temp_gainratio;
				chosen_attribute_name = i;
			}
		}
		else{ //连续
			multimap<double, string>::const_iterator index;
			temp_gainratio = FindDividingPoint(remain_Numerical_Attribute_Samples.at(*i), index);
			if (temp_gainratio > max_gainratio){
				max_gainratio = temp_gainratio;
				chosen_attribute_name = i;
				chosen_index = index;
			}
		}

	}
	current_node->attribute_name = *chosen_attribute_name;

	vector<map<string, string> > new_remain_Samples;
	vector<string> new_remain_Attribute_Names;
	//构建子节点的remain_Attribute_Names
	if (Attribute_Name_Values[*chosen_attribute_name][0].compare("ReaL")){ //离散
		for (i = remain_Attribute_Names.begin(); i != remain_Attribute_Names.end(); ++i){
			if ((*i).compare(*chosen_attribute_name))
				new_remain_Attribute_Names.push_back(*i);
		}
	}
	//根据确定的name生成相应value的子节点
	i = Attribute_Name_Values[*chosen_attribute_name].begin();
	if (i->compare("ReaL")){
		for (; i != Attribute_Name_Values[*chosen_attribute_name].end(); ++i){
			DecisionTreeNode* child_node = new DecisionTreeNode();
			child_node->attribute_pre_value = *i;
			for (vector<map<string, string> >::iterator j = remain_Samples.begin(); j != remain_Samples.end(); ++j){ //构建子节点的remain_Samples
				if (!(*i).compare(j->at(*chosen_attribute_name))){
					new_remain_Samples.push_back(*j);
				}
			}
			//递归生成子节点
			BuildContinuousDecisionTree(child_node, new_remain_Samples, new_remain_Attribute_Names, remain_Numerical_Attribute_Samples);
			current_node->children.push_back(child_node);
			new_remain_Samples.clear();
		}
	}
	else{ //连续
		int condition;
		map<string, multimap<double, string> > new_remain_Numerical_Attribute_Samples;
		vector<string> lr_attribute_value;
		lr_attribute_value = DivideContinuousIntoTwoParts(chosen_index);
		for (i = lr_attribute_value.begin(); i != lr_attribute_value.end(); ++i){
			DecisionTreeNode* child_node = new DecisionTreeNode();
			child_node->attribute_pre_value = *i;
			condition = UpdateContinuousSamples(remain_Samples, new_remain_Samples, new_remain_Numerical_Attribute_Samples, *chosen_attribute_name, *i);
			if (condition){ //若子节点的Samples为0或满直接返回多数决，防止无限递归
				current_node->attribute_name = current_node->majority_forecasting_attribute_value;
				return current_node;
			}
			BuildContinuousDecisionTree(child_node, new_remain_Samples, remain_Attribute_Names, new_remain_Numerical_Attribute_Samples);
			current_node->children.push_back(child_node);
			new_remain_Samples.clear();
			new_remain_Numerical_Attribute_Samples.clear();
		}
	}
	return current_node;
}

double C4d5::ComputeSplitEntropy(const vector<map<string, string> >& remain_Samples, string attribute_name, int& number)const{
	map<string, int> amount;
	number = 0;
	for (vector<map<string, string> >::const_iterator i = remain_Samples.begin(); i != remain_Samples.end(); ++i){
		++amount[i->at(attribute_name)];
		++number;
	}
	return Information(amount, number);
}

void C4d5::RecordEN(const vector<map<string, string> >& remain_Samples, ID3::DecisionTreeNode* current_node){
	current_node->majority_forecasting_attribute_value = MajorityDecision(remain_Samples);
	current_node->N = remain_Samples.size();
	if (!current_node->N){ //remain_Samples为空
		current_node->E = 0;
		return;
	}
	map<string, int> amount;
	int max = -1;
	for (vector<map<string, string> >::const_iterator i = remain_Samples.begin(); i != remain_Samples.end(); ++i){
		++amount[i->at(forecasting_attribute_name)];
	}
	for (map<string, int>::iterator i = amount.begin(); i != amount.end(); ++i){
		if (i->second > max){
			max = i->second;
		}
	}
	current_node->E = current_node->N - max;
}

double C4d5::ErrorRate(const int E, const int N, const double z)const{
	if (!N) return 0;
	double f = (double)E / N;
	double zz = z*z;
	return (f + zz / 2 / N + z*sqrt(f / N - f*f / N + zz / 4 / N / N)) / (1 + zz / N);
}

bool C4d5::SubtreeReplacement(ID3::DecisionTreeNode* current_node, const double z){
	if (current_node->children.empty()) return 0; //叶节点
	DecisionTreeNode* child_node;
	double current_node_error_rate, children_error_rate = 0;
	for (vector<DecisionTreeNode*>::iterator i = current_node->children.begin(); i != current_node->children.end(); ++i){
		child_node = *i;
		if (!(child_node->children.empty())) return 0; //子节点不为叶子节点
		children_error_rate += (child_node->N*ErrorRate(child_node->E, child_node->N, z));
	}
	children_error_rate /= current_node->N;
	current_node_error_rate = ErrorRate(current_node->E, current_node->N, z);
	if (children_error_rate >= current_node_error_rate){ //剪枝
		Pruning(current_node, z);
		return 1;
	}
	return 0;
}

void C4d5::Pruning(ID3::DecisionTreeNode* current_node, const double z){
	for (vector<DecisionTreeNode*>::iterator i = current_node->children.begin(); i != current_node->children.end(); ++i){
		delete *i;
	}
	current_node->children.clear();
	current_node->attribute_name = current_node->majority_forecasting_attribute_value;
}

bool C4d5::PessimisticPruning(ID3::DecisionTreeNode* current_node, const double z){
	bool is_pruned = 0;
	if (current_node->children.empty()) return 0;
	for (vector<DecisionTreeNode*>::iterator i = current_node->children.begin(); i != current_node->children.end(); ++i){
		is_pruned |= PessimisticPruning(*i, z);
		if (SubtreeReplacement(current_node, z)){
			return 1;
		}
	}
	return is_pruned;
}

void C4d5::PessimisticPruning(const double p){
	double z = ComputeZ(p);
	bool is_pruned;
	do{
		is_pruned = 0;
		is_pruned |= PessimisticPruning(root, z);
	} while (is_pruned);
}

double C4d5::ComputeZ(const double p)const{
	if (p == 0.25) return 0.675;
	if (p >= 0.5) return 0;
	double PI = 3.1415926;
	double z = 1 - 2 * p;
	return sqrt(PI / 2)*(z + PI / 12 * pow(z, 3) + 7 * pow(PI, 2) / 480 * pow(z, 5) + 127 * pow(PI, 3) / 40320 * pow(z, 7) + 4369 * pow(PI, 4) / 5806080 * pow(z, 9));
}

string C4d5::Predict(map<string, string> example)const{
	if (root == NULL){
		cout << "请先生成决策树" << std::endl;
		return "ERROR";
	}
	DecisionTreeNode* p = root;
	vector<ID3::DecisionTreeNode*>::const_iterator i = p->children.begin();
	while (i != p->children.end()){
		if ((!example[p->attribute_name].compare((*i)->attribute_pre_value)) || (IsInequalitySatisfied((*i)->attribute_pre_value, example[p->attribute_name]))){
			p = *i;
			i = p->children.begin();
			continue;
		}
		++i;
	}
	return p->attribute_name;
}

double C4d5::Predict()const{
	return ID3::Predict();
}

void C4d5::PreProcessor(int segments){
	if (Gethasreal()){
		vector<map<string, double> > Real_Samples; //实数的实例
		vector<map<string, double> > MinMaxDistance; //储存将实数分段的信息
		//将Samples中的实数属性存到Real_Samples中
		for (vector<map<string, string> >::iterator i = Samples.begin(); i != Samples.end(); ++i){
			map<string, double> real_item;
			for (map<string, string>::iterator j = i->begin(); j != i->end(); ++j){
				if (!Attribute_Name_Values[j->first][0].compare("ReaL")){
					real_item[j->first] = atof(j->second.c_str());
				}
			}
			Real_Samples.push_back(real_item);
		}
		//记录最大最小值
		MinMaxDistance.push_back(Real_Samples[0]);
		MinMaxDistance.push_back(Real_Samples[0]);
		MinMaxDistance.push_back(Real_Samples[0]);
		for (vector<map<string, double> >::iterator i = Real_Samples.begin() + 1; i != Real_Samples.end(); ++i){
			for (map<string, double>::iterator j = i->begin(); j != i->end(); ++j){
				if (j->second < MinMaxDistance[0][j->first]){
					MinMaxDistance[0][j->first] = j->second; //min
				}
				else if (j->second > MinMaxDistance[1][j->first]){
					MinMaxDistance[1][j->first] = j->second; //max
				}
				if (i == Real_Samples.end() - 1){
					MinMaxDistance[2][j->first] = (MinMaxDistance[1][j->first] - MinMaxDistance[0][j->first]) / (double)segments;
				}
			}
		}
		//将实数分段，更新Attribute_Name_Values
		stringstream ss_temp;
		for (map<string, double>::iterator i = MinMaxDistance[0].begin(); i != MinMaxDistance[0].end(); ++i){ //遍历实数的name
			Attribute_Name_Values[i->first].pop_back();
			double value_temp = i->second;
			for (int j = 0; j < segments; ++j){
				string value_l, value_h;
				ss_temp << value_temp;
				ss_temp >> value_l;
				ss_temp.clear();
				value_temp += MinMaxDistance[2][i->first];
				ss_temp << value_temp;
				ss_temp >> value_h;
				ss_temp.clear();
				Attribute_Name_Values[i->first].push_back(value_l + "-" + value_h);
			}
		}
		//更新Samples，将实数归类到已分好的段中
		for (vector<map<string, double> >::iterator i = Real_Samples.begin(); i != Real_Samples.end(); ++i){ //行，实例
			for (map<string, double>::iterator j = i->begin(); j != i->end(); ++j){ //列，属性名
				int k;
				for (k = 0; j->second > MinMaxDistance[0][j->first] + MinMaxDistance[2][j->first] * (k + 1); ++k);
				Samples[i - Real_Samples.begin()][j->first] = Attribute_Name_Values[j->first][k];
			}
		}
		forecasting_attribute_value_length = Attribute_Name_Values[forecasting_attribute_name].size();
		Sethasreal(0);
	}
}

void C4d5::test(){
	using std::endl;
	map<string, multimap<double, string> > Numerical_Attribute_Samples; //存放连续属性 属性名，multimap<数值，类别>
	PreProcessorForContinuous(Samples, Numerical_Attribute_Samples);
	multimap<double, string>::iterator i;
//	FindDividingPoint(Numerical_Attribute_Samples[Attribute_Names[0]], i);
}
