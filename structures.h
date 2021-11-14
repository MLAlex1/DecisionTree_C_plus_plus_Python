#pragma once

#include <iostream>
#include <vector>
#include <map>

using namespace std;

class Node {
public:
	bool isleaf = false;
	size_t label;
	double attrValue;
	vector<Node> children;
	int col = -1;
	double gini;
	size_t n_samples;
	map <int, int> class_count;
	vector<vector<double>> data;
};

struct Ginis {
	double gini_left, gini_right, gini_left_nonstand, gini_right_nonstand;
};

struct MajorLabel {
	size_t major;
	std::map <int, int> class_count;
};

struct SplitInfo {
	int idx;
	double value, gini, gini_left, gini_right;;

};


struct DataSplits {
	std::vector<std::vector<double>> data_left, data_right;
};
