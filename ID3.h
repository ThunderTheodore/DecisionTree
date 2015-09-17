/*
 * ID3.h
 *
 *  Created on: Mar 19, 2015
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
	int segments, minLeaf; //segmentsÎªÊµÊýÔ€ŽŠÀíÊ±·ÖÎªŒž¶Î£»minLeafÎªÒ¶×Ó×îÐ¡ÊµÀýÊý
	string relation;
	string forecasting_attribute_name; //ŒÇÂŒÐèÒªÔ€²âµÄÊôÐÔÃû
	int forecasting_attribute_value_length; //ŒÇÂŒÐèÒªÔ€²âµÄÊôÐÔµÄ·ÖÖ§Êý
	map<string, vector<string> > Attribute_Name_Values; //ŒÇÂŒÊôÐÔÃûºÍ¶ÔÓŠµÄÊôÐÔÖµ
	vector<string> Attribute_Names; //ŒÇÂŒÊôÐÔÃû
	vector<map<string, string> > Samples; //ÓÃÀŽÑµÁ·µÄÊµÀý
	vector<map<string, string> > Test_Samples; //ÓÃÀŽ²âÊÔµÄÊµÀý
	struct DecisionTreeNode{
		string attribute_name;
		string attribute_pre_value;
		int E; int N; //ÓÃÓÚ±¯¹ÛŒôÖŠEŽíÎóÊµÀýÊý£»Nµ±Ç°×ÜÊµÀýÊý
		string majority_forecasting_attribute_value; //µ±Ç°Ê£ÓàÊµÀýÖÐ×î¶àµÄforecastingÊôÐÔÖµ
		vector<DecisionTreeNode*> children;
		DecisionTreeNode() :attribute_name(""), attribute_pre_value(""), majority_forecasting_attribute_value(""), E(-1), N(-1){};
	} *root;

	//ÊµÊýµÄÔ€ŽŠÀí
	void PreProcessor();

	//ìØÏà¹Øº¯Êý
	//ŒÆËãinfo([1,2,3,...])£¬numberŒÇÂŒÊµÀýžöÊý
	double Information(map<string, int>& amount, int number)const;
	//ÇóÊôÐÔÃûÏÂ·ÖÖ§µÄÐÅÏ¢Öµ£»µ±Ç°¿ŒÂÇµÄÊµÀýŒ¯£¬µ±Ç°¿ŒÂÇµÄÊôÐÔÃû£¬ÊôÐÔÃûµÄÊôÐÔÖµ·ÖÖ§
	double ComputeEntropy(vector<map<string, string> >& remain_Samples, string attribute_name, string attribute_value, int& number)const;
	//Çóµ±Ç°ÊµÀýŒ¯µÄÐÅÏ¢Öµ£¬²¢žüÐÂEºÍN
	double ComputeEntropy(vector<map<string, string> >& remain_Samples, int& number)const;
	//ÊôÐÔÃû¶ÔÓŠµÄÐÅÏ¢ÔöÒæ
	double ComputeGain(vector<map<string, string> >& remain_Samples, string attribute_name);

	//Éú³ÉŸö²ßÊ÷µÄÖÕÖ¹ÌõŒþº¯Êý
	//ÅÐ¶ÏremainÖÐforecastingÊôÐÔÖµÊÇ·ñ¶ŒÏàÍ¬£¬Èô·ñ·µ»Ø"ImpurE"
	string IsPure(vector<map<string, string> >& remain_Samples)const;
	//·µ»ØremainÖÐ×î¶àµÄforecastingÊôÐÔÖµ
	string MajorityDecision(vector<map<string, string> >& remain_Samples)const;
	//ÏÈÐòµÝ¹é¹¹œšŸö²ßÊ÷
	virtual DecisionTreeNode* BuildDecisionTree(DecisionTreeNode* current_node, vector<map<string, string> > remain_Samples, vector<string> remain_Attribute_Names);
	//ÏÈÐòµÝ¹éŽòÓ¡Ÿö²ßÊ÷
	void PrintDecisionTree(DecisionTreeNode* node, int depth)const;
	//ºóÐòµÝ¹éÊÍ·ÅŸö²ßÊ÷
	void FreeDecisionTree(DecisionTreeNode* node);
public:
	//ÓÉÓÚarff²¢²»×¢Ã÷Ô€²âÄÄžöÊôÐÔ£¬²»·ÁÄ¬ÈÏÎª×îºóÒ»žöÊôÐÔ
	ID3(string path, int segments = 6, int minLeaf = 2);
	void PrintDecisionTree()const;
	void BuildDecisionTree();
	//°ŽÕÕ°Ù·Ö±Èœ«Samples·ÖÎªÑµÁ·ºÍ²âÊÔµÄŒ¯ºÏ
	void BuildDecisionTree(double percentage);
	//žùŸÝžø¶šµÄµ¥žöÑùÀýœøÐÐÔ€²â
	string Predict(map<string,string> example)const;
	//žùŸÝpercentage»®·ÖµÄ²âÊÔŒ¯œøÐÐ²âÊÔ£¬·µ»Ø×ŒÈ·ÂÊ
	double Predict();
	virtual void PessimisticPruning(double z = 0.69){};
	void test();
	virtual ~ID3();
};

#endif /* ID3_H_ */
