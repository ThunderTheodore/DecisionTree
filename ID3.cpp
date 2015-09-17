/*
 * ID3.cpp
 *
 *  Created on: Mar 19, 2015
 *      Author: th
 */
#include "ID3.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <math.h>

using std::ifstream;
using std::ios;
using std::cout;
using std::endl;
using std::getline;
using std::stringstream;

ID3::ID3(string path, int segments, int minLeaf){
	ifstream in;
	string source;
	bool data = 0;
	const string label1 = "@relation ";
	const string label2 = "@attribute ";
	const string label3 = "@data";
	has_real = 0;
	this->root = new DecisionTreeNode();
	this->segments = segments, this->minLeaf = minLeaf;
	relation = "";
	in.open(path.c_str(), ios::in);
	if (!in.is_open()) cout << "Â·Ÿ¶ŽíÎó";
	while (!in.eof()){
		getline(in, source);
		if (!label1.compare(source.substr(0, 10))) //@relation
			relation = source.substr(10, source.length() - 10);
		else if (!label2.compare(source.substr(0, 11))){ //@attribute
			int i = 11, j;
			while (source[i] != ' ') ++i;
			Attribute_Names.push_back(source.substr(11, i - 11));
			forecasting_attribute_name = source.substr(11, i - 11);
			if (source[i + 1] == '{') i += 2;
			else if (source.substr(i + 1, 4) == "real"){ //ÊµÊý
				Attribute_Name_Values[Attribute_Names.back()].push_back("ReaL");
				has_real = 1;
				continue;
			}
			else{
				cout << "error";
				break;
			}
			for (j = i; source[j] != '}'; ++j){
				if (source[j] == ','){
					Attribute_Name_Values[Attribute_Names.back()].push_back(source.substr(i, j - i));
					i = j + 1;
				}
			}
			Attribute_Name_Values[Attribute_Names.back()].push_back(source.substr(i, j - i));
		}
		else if (!label3.compare(source.substr(0, 5))){ //@data
			data = 1;
		}
		else if (data){
			map<string, string> item;
			int i, j, k;
			for (i = 0, j = 0, k = 0; source[j] != '\0'; ++j){
				if (source[j] == ','){
					item[Attribute_Names[k++]] = source.substr(i, j - i);
					i = j + 1;
				}
			}
			item[Attribute_Names[k]] = source.substr(i, j - i);
			Samples.push_back(item);
		}
	}
	forecasting_attribute_value_length = Attribute_Name_Values[forecasting_attribute_name].size();
}

void ID3::PreProcessor(){
	if (has_real){
		vector<map<string, double> > Real_Samples; //ÊµÊýµÄÊµÀý
		vector<map<string, double> > MinMaxDistance; //Ž¢Žæœ«ÊµÊý·Ö¶ÎµÄÐÅÏ¢
		//œ«SamplesÖÐµÄÊµÊýÊôÐÔŽæµœReal_SamplesÖÐ
		for (vector<map<string, string> >::iterator i = Samples.begin(); i != Samples.end(); ++i){
			map<string, double> real_item;
			for (map<string, string>::iterator j = i->begin(); j != i->end(); ++j){
				if (!Attribute_Name_Values[j->first][0].compare("ReaL")){
					real_item[j->first] = atof(j->second.c_str());
				}
			}
			Real_Samples.push_back(real_item);
		}
		//ŒÇÂŒ×îŽó×îÐ¡Öµ
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
		//œ«ÊµÊý·Ö¶Î£¬žüÐÂAttribute_Name_Values
		stringstream ss_temp;
		for (map<string, double>::iterator i = MinMaxDistance[0].begin(); i != MinMaxDistance[0].end(); ++i){ //±éÀúÊµÊýµÄname
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
		//žüÐÂSamples£¬œ«ÊµÊý¹éÀàµœÒÑ·ÖºÃµÄ¶ÎÖÐ
		for (vector<map<string, double> >::iterator i = Real_Samples.begin(); i != Real_Samples.end(); ++i){ //ÐÐ£¬ÊµÀý
			for (map<string, double>::iterator j = i->begin(); j != i->end(); ++j){ //ÁÐ£¬ÊôÐÔÃû
				int k;
				for (k = 0; j->second > MinMaxDistance[0][j->first] + MinMaxDistance[2][j->first] * (k + 1); ++k);
				Samples[i - Real_Samples.begin()][j->first] = Attribute_Name_Values[j->first][k];
			}
		}
		forecasting_attribute_value_length = Attribute_Name_Values[forecasting_attribute_name].size();
	}
}

double ID3::Information(map<string, int>& amount, int number)const{
	double ans = 0;
	for (map<string, int>::iterator i = amount.begin(); i != amount.end(); ++i){
		ans += (-i->second / (double)number*log(i->second / (double)number) / log(2.0));
	}
	return ans;
}

double ID3::ComputeEntropy(vector<map<string, string> >& remain_Samples, string attribute_name, string attribute_value, int& number)const{
	map<string, int> amount;
	number = 0;
	for (vector<map<string, string> >::iterator i = remain_Samples.begin(); i != remain_Samples.end(); ++i){
		if (!attribute_value.compare(i->at(attribute_name))){
			++amount[i->at(forecasting_attribute_name)];
			++number;
		}
	}
	return Information(amount, number);
}

double ID3::ComputeEntropy(vector<map<string, string> >& remain_Samples, int& number)const{
	map<string, int> amount;
	number = 0;
	for (vector<map<string, string> >::iterator i = remain_Samples.begin(); i != remain_Samples.end(); ++i){
		++amount[i->at(forecasting_attribute_name)];
		++number;
	}
	return Information(amount, number);
}

double ID3::ComputeGain(vector<map<string, string> >& remain_Samples, string attribute_name){
	int sum, amount; //sumÎªÊµÀý×ÜÊý£¬amountÎªÊôÐÔÖµÏÂÊµÀýžöÊý
	double parent_entropy = ComputeEntropy(remain_Samples, sum);
	double children_entropy = 0;
	for (unsigned int i = 0; i < Attribute_Name_Values[attribute_name].size(); ++i){
		double temp = ComputeEntropy(remain_Samples, attribute_name, Attribute_Name_Values[attribute_name][i], amount);
		children_entropy += (double)amount / sum*temp;
	}
	return parent_entropy - children_entropy;
}

string ID3::IsPure(vector<map<string, string> >& remain_Samples)const{
	string forecasting_attribute_value = remain_Samples[0][forecasting_attribute_name];
	for (vector<map<string, string> >::iterator i = remain_Samples.begin() + 1; i != remain_Samples.end(); ++i){
		if (forecasting_attribute_value.compare(i->at(forecasting_attribute_name)))
			return "ImpurE";
	}
	return forecasting_attribute_value;
}

string ID3::MajorityDecision(vector<map<string, string> >& remain_Samples)const{
	if (remain_Samples.empty()) return "unknown";
	map<string, int> count;
	map<string, int>::iterator ans;
	int max = 0;
	for (vector<map<string, string> >::iterator i = remain_Samples.begin(); i != remain_Samples.end(); ++i){
		++count[i->at(forecasting_attribute_name)];
	}
	for (map<string, int>::iterator i = count.begin(); i != count.end(); ++i){
		if ((*i).second > max){
			max = (*i).second;
			ans = i;
		}
	}
	return (*ans).first;
}

ID3::DecisionTreeNode* ID3::BuildDecisionTree(DecisionTreeNode* current_node, vector<map<string, string> > remain_Samples, vector<string> remain_Attribute_Names){
	//ÖÕÖ¹ÌõŒþ
	if (remain_Samples.size() <= minLeaf){ //Ê£ÓàÊµÀýÊýÁ¿Ð¡ÓÚãÐÖµ
		current_node->attribute_name = MajorityDecision(remain_Samples);
		return current_node;
	}
	string pure = IsPure(remain_Samples);
	if (pure.compare("ImpurE")){ //ÐÅÏ¢ÖµÎª0
		current_node->attribute_name = pure;
		return current_node;
	}
	if (remain_Attribute_Names.empty()){ //Ã»ÓÐÊ£ÓàµÄÊôÐÔÃû
		current_node->attribute_name = MajorityDecision(remain_Samples);
		return current_node;
	}
	//ÕÒ×îŽóÐÅÏ¢ÔöÒæ
	double max_gain = -1, temp_gain;
	vector<string>::iterator chosen_attribute_name;
	vector<string>::iterator i;
	for (i = remain_Attribute_Names.begin(); i != remain_Attribute_Names.end(); ++i){
		temp_gain = ComputeGain(remain_Samples, *i);
		if (temp_gain > max_gain){
			max_gain = temp_gain;
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

void ID3::BuildDecisionTree(){
	vector<string> current_Attribute_Names;
	//²»¿ŒÂÇforecastingµÄÊôÐÔÃû
	PreProcessor();
	current_Attribute_Names.insert(current_Attribute_Names.begin(), Attribute_Names.begin(), Attribute_Names.end() - 1);
	root = BuildDecisionTree(root, Samples, current_Attribute_Names);
}

void ID3::BuildDecisionTree(double percentage){
	if (percentage < 0 || percentage>1){
		cout << "BuildDecisionTree(double percentage)²ÎÊýŽíÎó";
		return;
	}
	vector<string> current_Attribute_Names;
	PreProcessor();
	current_Attribute_Names.insert(current_Attribute_Names.begin(), Attribute_Names.begin(), Attribute_Names.end() - 1);
	int training_size = (int)((1.0 - percentage)*Samples.size());
	int test_size = Samples.size() - training_size;
	Test_Samples.insert(Test_Samples.begin(), Samples.end() - test_size, Samples.end()); //ºópercentage±ÈÀýµÄÊµÀý×÷Îª²âÊÔ
	Samples.erase(Samples.end() - test_size, Samples.end());
	root = BuildDecisionTree(root, Samples, current_Attribute_Names);
}

void ID3::PrintDecisionTree(ID3::DecisionTreeNode* node, int depth)const{
	int i;
	for (i = 0; i < depth; ++i) cout << '\t';
	if (!node->attribute_pre_value.empty()){
		cout << node->attribute_pre_value <<" "<<node->E<<" "<<node->N<< endl;
		for (i = 0; i < depth + 1; ++i) cout << '\t';
	}
	else cout << '\t';
	cout << node->attribute_name << endl;
	for (unsigned i = 0; i < node->children.size(); ++i){
		PrintDecisionTree(node->children[i], depth + 2);
	}
}

void ID3::PrintDecisionTree()const{
	PrintDecisionTree(root, 0);
}

void ID3::FreeDecisionTree(ID3::DecisionTreeNode* node){
	if (node == NULL) return;
	for (vector<DecisionTreeNode*>::iterator i = node->children.begin(); i != node->children.end(); ++i){
		FreeDecisionTree(*i);
	}
	delete node;
}

ID3::~ID3(){
	FreeDecisionTree(root);
}

string ID3::Predict(map<string, string> example)const{
	if (root == NULL){
		cout << "ÇëÏÈÉú³ÉŸö²ßÊ÷" << endl;
		return "ERROR";
	}
	DecisionTreeNode* p = root;
	vector<ID3::DecisionTreeNode*>::iterator i = p->children.begin();
	while (i != p->children.end()){
		if (!example[p->attribute_name].compare((*i)->attribute_pre_value)){
			p = *i;
			i = p->children.begin();
			continue;
		}
		++i;
	}
	return p->attribute_name;
}

double ID3::Predict(){
	if (Test_Samples.empty()){
		cout << "²âÊÔŒ¯Îª¿Õ£¬ÇëÓÃBuildDecisionTree(double percentage)¹¹Ôì";
		return 0;
	}
	int correct = 0;
	for (vector<map<string, string> >::iterator i = Test_Samples.begin(); i != Test_Samples.end(); ++i){
		if (!(i->at(forecasting_attribute_name)).compare(Predict(*i))){
			++correct;
		}
	}
	return (double)correct / Test_Samples.size();
}

void ID3::test(){
	vector<map<string, string> > remain_Samples(Samples);
	//	cout << ComputeGainRatio(remain_Samples, "windy");
}



