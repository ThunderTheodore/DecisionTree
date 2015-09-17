/*
 * C4d5.cpp
 *
 *  Created on: Mar 19, 2015
 *      Author: th
 */
#include "C4d5.h"

#include <iostream>
#include <math.h>
using std::cout;

double C4d5::ComputeGainRatio(vector<map<string, string> >& remain_Samples, string attribute_name){
	int number;
	double temp = ComputeSplitEntropy(remain_Samples, attribute_name, number);
	if (!temp) return 65535; //·ÖÁÑÐÅÏ¢ÖµÎª0Ê±·µ»Ø×îŽóÖµ
	else return ComputeGain(remain_Samples, attribute_name) / temp;
}

ID3::DecisionTreeNode* C4d5::BuildDecisionTree(DecisionTreeNode* current_node, vector<map<string, string> > remain_Samples, vector<string> remain_Attribute_Names){
	//ŒÆËãEºÍN
	RecordEN(remain_Samples, current_node);
	//ÖÕÖ¹ÌõŒþ
	if (remain_Samples.size() <= minLeaf){ //Ê£ÓàÊµÀýÊýÁ¿Ð¡ÓÚãÐÖµ
		current_node->attribute_name = current_node->majority_forecasting_attribute_value;
		return current_node;
	}
	string pure = IsPure(remain_Samples);
	if (pure.compare("ImpurE")){ //ÐÅÏ¢ÖµÎª0
		current_node->attribute_name = pure;
		return current_node;
	}
	if (remain_Attribute_Names.empty()){ //Ã»ÓÐÊ£ÓàµÄÊôÐÔÃû
		current_node->attribute_name = current_node->majority_forecasting_attribute_value;
		return current_node;
	}
	//ÕÒ×îŽóÐÅÏ¢ÔöÒæ
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
	//¹¹œš×ÓœÚµãµÄremain_Attribute_Names
	vector<map<string, string> > new_remain_Samples;
	vector<string> new_remain_Attribute_Names;
	for (i = remain_Attribute_Names.begin(); i != remain_Attribute_Names.end(); ++i){
		if ((*i).compare(*chosen_attribute_name))
			new_remain_Attribute_Names.push_back(*i);
	}
	//žùŸÝÈ·¶šµÄnameÉú³ÉÏàÓŠvalueµÄ×ÓœÚµã
	for (i = Attribute_Name_Values[*chosen_attribute_name].begin(); i != Attribute_Name_Values[*chosen_attribute_name].end(); ++i){
		DecisionTreeNode* child_node = new DecisionTreeNode();
		child_node->attribute_pre_value = *i;
		for (vector<map<string, string> >::iterator j = remain_Samples.begin(); j != remain_Samples.end(); ++j){ //¹¹œš×ÓœÚµãµÄremain_Samples
			if (!(*i).compare(j->at(*chosen_attribute_name))){
				new_remain_Samples.push_back(*j);
			}
		}
		//µÝ¹éÉú³É×ÓœÚµã
		BuildDecisionTree(child_node, new_remain_Samples, new_remain_Attribute_Names);
		current_node->children.push_back(child_node);
		new_remain_Samples.clear();
	}
	return current_node;
}

double C4d5::ComputeSplitEntropy(vector<map<string, string> >& remain_Samples, string attribute_name, int& number)const{
	map<string, int> amount;
	number = 0;
	for (vector<map<string, string> >::iterator i = remain_Samples.begin(); i != remain_Samples.end(); ++i){
		++amount[(*i)[attribute_name]];
		++number;
	}
	return Information(amount, number);
}

void C4d5::RecordEN(vector<map<string, string> >& remain_Samples, ID3::DecisionTreeNode* current_node){
	current_node->majority_forecasting_attribute_value = MajorityDecision(remain_Samples);
	current_node->N = remain_Samples.size();
	if (!current_node->N){ //remain_SamplesÎª¿Õ
		current_node->E = 0;
		return;
	}
	map<string, int> amount;
	int max = -1;
	for (vector<map<string, string> >::iterator i = remain_Samples.begin(); i != remain_Samples.end(); ++i){
		++amount[i->at(forecasting_attribute_name)];
	}
	for (map<string, int>::iterator i = amount.begin(); i != amount.end(); ++i){
		if (i->second > max){
			max = i->second;
		}
	}
	current_node->E = current_node->N - max;
}

double C4d5::ErrorRate(int E, int N, double z)const{
	if (!N) return 0;
	double f = (double)E / N;
	double zz = z*z;
	return (f + zz / 2 / N + z*sqrt(f / N - f*f / N + zz / 4 / N / N)) / (1 + zz / N);
}

bool C4d5::SubtreeReplacement(ID3::DecisionTreeNode* current_node,double z){
	if (current_node->children.empty()) return 0; //Ò¶œÚµã
	DecisionTreeNode* child_node;
	double current_node_error_rate, children_error_rate = 0;
	for (vector<DecisionTreeNode*>::iterator i = current_node->children.begin(); i != current_node->children.end(); ++i){
		child_node = *i;
		if (!(child_node->children.empty())) return 0; //×ÓœÚµã²»ÎªÒ¶×ÓœÚµã
		children_error_rate += (child_node->N*ErrorRate(child_node->E, child_node->N, z));
	}
	children_error_rate /= current_node->N;
	current_node_error_rate = ErrorRate(current_node->E, current_node->N, z);
	if (children_error_rate >= current_node_error_rate){ //ŒôÖŠ
		Pruning(current_node, z);
		return 1;
	}
	return 0;
}

void C4d5::Pruning(ID3::DecisionTreeNode* current_node, double z){
	for (vector<DecisionTreeNode*>::iterator i = current_node->children.begin(); i != current_node->children.end(); ++i){
		delete *i;
	}
	current_node->children.clear();
	current_node->attribute_name = current_node->majority_forecasting_attribute_value;
}

bool C4d5::PessimisticPruning(ID3::DecisionTreeNode* current_node, double z){
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

void C4d5::PessimisticPruning(double p){
	double z = ComputeZ(p);
	bool is_pruned;
	do{
		is_pruned = 0;
		is_pruned |= PessimisticPruning(root, z);
	} while (is_pruned);
}

double C4d5::ComputeZ(double p)const{
	if (p == 0.25) return 0.675;
	if (p >= 0.5) return 0;
	double PI = 3.1415926;
	double z = 1 - 2 * p;
	return sqrt(PI / 2)*(z + PI / 12 * pow(z, 3) + 7 * pow(PI, 2) / 480 * pow(z, 5) + 127 * pow(PI, 3) / 40320 * pow(z, 7) + 4369 * pow(PI, 4) / 5806080 * pow(z, 9));
}

void C4d5::test(){
	vector<map<string, string> > remain_Samples(Samples);
	cout << ComputeGainRatio(remain_Samples, "outlook");
}



