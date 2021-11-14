#include "utilities.h"

vector<double>* sort_column(vector<vector<double>> data, size_t idx) {
	// we sort the chosen column to compute the thresholds 
	// given by the average between previous and following value

	vector<double> *col_sorted = new vector<double>;
	for (auto row : data) {
		col_sorted->push_back(row[idx]);
	}
	if (col_sorted->size() < 2) {
		return col_sorted;
	}
	sort(col_sorted->begin(), col_sorted->end());
	vector<double>* col_averages = new vector<double>;
	for (int i = 0; i < col_sorted->size() - 1; i++) {
		col_averages->push_back(((*col_sorted)[i] + (*col_sorted)[i + 1]) / 2);
	}

	return col_averages;
}


MajorLabel get_major_label(vector<vector<double>> data, size_t lbl) {
	// major label in the data

	map <int, int> class_count;
	for (auto row : data) {
		int datacls = (int)row[lbl];
		if (class_count.find(datacls) == class_count.end()) {
			class_count[datacls] = 1;
		}
		else {
			class_count[datacls] ++;
		}
	}
	int largest_val = -1;
	size_t idx = 0;

	for (map<int, int>::iterator iter = class_count.begin() ; iter != class_count.end(); iter++) {
		if (iter->second > largest_val) {
			idx = iter->first;
			largest_val = iter->second;
		}
	}

	return { idx , class_count };
}


size_t dfs(Node node, vector<double> row, int idx) {
	// perform dfs

	if (node.isleaf) {
		return node.label;
	}

	if (row[idx] <= node.attrValue) {
		return dfs(node.children[0], row, node.children[0].col);
	}
	else {
		return dfs(node.children[1], row, node.children[1].col);
	}
}


void dfs_print(Node root, int depth) {
	// print tree 

	if (root.isleaf) {
		cout << string(depth, ' ') << "leaf" << " gini: " << root.gini << " label: " << root.label << endl;
		return;
	}
	else {
		cout << endl;
		cout << string(depth, ' ') << "X" << root.col << "<" << root.attrValue << " gini: " << root.gini << " label: " << root.label << endl;
		dfs_print(root.children[0], depth + 3);
		dfs_print(root.children[1], depth + 3);
	}
}
