/*
 * C4d5.h
 *
 *  Created on: Mar 19, 2015
 *      Author: th
 */

#ifndef C4D5_H_
#define C4D5_H_

#include "ID3.h"

class C4d5 :public ID3{
protected:
	//C4.5Çó·ÖÁÑÐÅÏ¢¶ÈÁ¿
	double ComputeSplitEntropy(vector<map<string, string> >& remain_Samples, string attribute_name, int& number)const;
	//C4.5ÇóÐÅÏ¢ÔöÒæ
	double ComputeGainRatio(vector<map<string, string> >& remain_Samples, string attribute_name);
	DecisionTreeNode* BuildDecisionTree(DecisionTreeNode* current_node, vector<map<string, string> > remain_Samples, vector<string> remain_Attribute_Names);
	//ŒÇÂŒœáµãÖÐµÄEºÍNºÍmajority_forecasting_attribute_value
	void RecordEN(vector<map<string, string> >& remain_Samples, DecisionTreeNode* current_node);
	//ŒÆËã±¯¹ÛµÄÎó²îÂÊ£¬ÈôNÎª0·µ»Ø0
	double ErrorRate(int E, int N, double z)const;
	//×ÓÊ÷ÖÃ»»£¬Èô·¢ÉúÌæ»»·µ»Ø1
	bool SubtreeReplacement(DecisionTreeNode* current_node, double z);
	//ŒôÖŠ£¬ÓÃµ±Ç°œÚµãÏÂµÄÒ¶×ÓŽúÌæµ±Ç°œÚµã
	void Pruning(DecisionTreeNode* current_node, double z);
	//ÏÈÐò±éÀúÊ÷Ñ°ÕÒÐèÒªŒôÖŠµÄœÚµã
	bool PessimisticPruning(DecisionTreeNode* current_node, double z);
	//žùŸÝÖÃÐÅ¶ÈcÇó±ê×ŒÕýÌ¬·Ö²Œµ¥²àzÖµ
	double ComputeZ(double p)const;
public:
	C4d5(string path, int segments = 6, int minLeaf = 2) :ID3(path, segments, minLeaf){};
	//ŒôÖŠº¯Êý
	void PessimisticPruning(double p = 0.25);
	void test();
};

#endif /* C4D5_H_ */
