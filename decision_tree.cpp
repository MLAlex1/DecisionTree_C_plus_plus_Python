#include "utilities.h"
#include "structures.h"

using namespace std;

double _getGini(vector<vector<double>> data, size_t lbl) {
	// gini for single data split

	map <int, int> class_count;
	size_t size = data.size();
	if (size == 0)
		return 0;

	for (auto row : data) {

		int datacls = (int)row[lbl];
		if (class_count.find(datacls) == class_count.end()) {
			class_count[datacls] = 1;
		}
		else {
			class_count[datacls] ++;
		}
	}

	double score = 0;
	double p = 0;
	for (auto i : class_count) {
		p = (double)i.second / (double)size;
		score += p * p;
	}

	return 1.0 - score;
};

Ginis getGini(vector<vector<double>> data_left, vector<vector<double>> data_right, size_t lbl) {
	// get gini for left and right data splits

	size_t n_instances = data_left.size() + data_right.size();

	double gini_nonstand;
	gini_nonstand = _getGini(data_left, lbl);
	double gini_left = gini_nonstand * (double)data_left.size() / (double)n_instances;
	double gini_left_nonst = gini_nonstand;

	gini_nonstand = _getGini(data_right, lbl);
	double gini_right = gini_nonstand * (double)data_right.size() / (double)n_instances;

	return { gini_left, gini_right, gini_left_nonst, gini_nonstand };

}


DataSplits left_right_data(size_t idx, double value, vector<vector<double>> data) {
	// split data into right and left datasets

	DataSplits data_split;
	for (auto i : data) {
		if (i[idx] <= value) {
			data_split.data_left.push_back(i);
		}
		else {
			data_split.data_right.push_back(i);
		}
	}

	return data_split;
}



SplitInfo best_split_info(vector<vector<double>> data, size_t n_classes, int min_size, size_t lbl) {
	// get info of the best split

	DataSplits groups;
	double b_score = 1, b_value = -1;
	int b_index = -1;
	Ginis gini_struct;
	double gini;
	double gini_left = 0, gini_right = 0, gini_left_nonst = 0, gini_right_nonst = 0;

	vector<double>* col_sorted;

	for (int idx = 0; idx <= n_classes; idx++) {
		col_sorted = sort_column(data, idx);
		for (auto col : *col_sorted) {
			groups = left_right_data(idx, col, data);

			// Reject if min_size is not guaranteed
			if (groups.data_left.size() < min_size || groups.data_right.size() < min_size)
				continue;
			gini_struct = getGini(groups.data_left, groups.data_right, lbl);
			gini = gini_struct.gini_left + gini_struct.gini_right;

			if (gini <= b_score) {
				b_score = gini;
				b_value = col;
				b_index = idx;
				// gini_left = gini_struct.gini_left;
				// gini_right = gini_struct.gini_right;
				gini_left_nonst = gini_struct.gini_left_nonstand;
				gini_right_nonst = gini_struct.gini_right_nonstand;
			}
		}

		if (b_index == -1) {
			// if we cannot avoid a split without violating min_size, 
			// we assign last value in sorted vector to b_value
			b_index = (int)col_sorted->size() - 1;
			b_value = (*col_sorted)[b_index];
		}
	}

	return { b_index, b_value, b_score, gini_left_nonst, gini_right_nonst };
}

Node assign_values_node(Node& node, vector<vector<double>> data, bool leaf, double gini, size_t lbl) {
	MajorLabel majorLbl;
	majorLbl = get_major_label(data, lbl);
	node.isleaf = leaf;
	node.label = majorLbl.major;
	node.class_count = majorLbl.class_count;
	node.n_samples = data.size();
	node.data = data;
	node.gini = gini;
	return node;
}


Node split(Node next_node, vector<vector<double>> data, int max_depth, int min_size, int depth = 0, size_t n_classes = 3, size_t lbl = 4) {
	// 

	SplitInfo outinfo;
	DataSplits groups;
	MajorLabel majorLbl;

	// check data not all same class, if so leaf
	majorLbl = get_major_label(data, lbl);               
	if (majorLbl.class_count.size() == 1) {
		next_node = assign_values_node(next_node, data, true, 0, lbl);
		return next_node;
	}

	outinfo = best_split_info(data, n_classes, min_size, lbl);
	groups = left_right_data(outinfo.idx, outinfo.value, data);
	
	if (groups.data_left.size() == 0) {
		next_node = assign_values_node(next_node, data, true, outinfo.gini_left, lbl);
		return next_node;
	}
	else if (groups.data_right.size() == 0) {
		next_node = assign_values_node(next_node, data, true, outinfo.gini_right, lbl);
		return next_node;
	}

	if (groups.data_left.size() <= min_size || groups.data_right.size() <= min_size || depth + 1 >= max_depth) {

		Node node_left;
		node_left = assign_values_node(node_left, groups.data_left, true, outinfo.gini_left, lbl);
		next_node.children.push_back(node_left);

		Node node_right;
		node_right = assign_values_node(node_right, groups.data_right, true, outinfo.gini_right, lbl);
		next_node.children.push_back(node_right);
		next_node = assign_values_node(next_node, data, false, 0, lbl);

		next_node.attrValue = outinfo.value;
		next_node.col = outinfo.idx;
	} 
	else {

		next_node.attrValue = outinfo.value;
		next_node.col = outinfo.idx;
		next_node = assign_values_node(next_node, data, false, next_node.gini, lbl);

		// left node
		Node empty_node;
		Node next_left = split(empty_node, groups.data_left, max_depth, min_size, depth + 1);
		next_left.gini = outinfo.gini_left;

		next_node.children.push_back(next_left);
		// right node
		Node empty_node1;
		Node next_right = split(empty_node1, groups.data_right, max_depth, min_size, depth + 1);
		next_right.gini = outinfo.gini_right;
		next_node.children.push_back(next_right);
	}

	return next_node;
}



class DecisionTree {

public:
	int max_depth, min_size;
	Node root_init;
	DecisionTree(int md, int ms) { max_depth = md, min_size = ms; };
	void fit(double **arr, int n_rows, int n_cols, bool printTree);
	size_t* predict_data(double **arr, int n_rows, int n_cols);
};

void DecisionTree::fit(double **arr, int n_rows, int n_cols, bool printTree) {

	int depth = 0;
	vector<vector<double>> data_numpy;

	for (int i = 0; i<n_rows; i++){
		vector<double> row;
		for (int j = 0; j<n_cols; j++){
			row.push_back(arr[i][j]);
		}
		data_numpy.push_back(row);
	}

	size_t lbl = data_numpy[0].size() - 1;
	size_t n_classes = get_major_label(data_numpy, lbl).class_count.size();

	Node root;
	double gini_nonstand;
	size_t n_instances = data_numpy.size();
	gini_nonstand = _getGini(data_numpy, lbl);
	double gini = gini_nonstand * (double)data_numpy.size() / (double)n_instances;
	root.gini = gini_nonstand;

	root = split(root, data_numpy, max_depth, min_size, depth, n_classes, lbl);

	if (printTree)
		dfs_print(root, 0);

	root_init = root;
}

size_t* DecisionTree::predict_data(double **arr, int n_rows, int n_cols) {
	
	vector<vector<double>> data;

	for (int i = 0; i<n_rows; i++){
		vector<double> row;
		for (int j = 0; j<n_cols; j++){
			row.push_back(arr[i][j]);
		}
		data.push_back(row);
	}

	size_t* out_preds = new size_t[data.size()];

	int idx = 0;
	for (auto row : data) {
		out_preds[idx] = dfs(root_init, row, root_init.col);
		idx++;
	}
	return out_preds;
}

extern "C" {
	__declspec(dllexport) DecisionTree* new_tree(int max_d, int min_s) {
		DecisionTree *dc = new DecisionTree(max_d, min_s);
		return dc;
	}
	__declspec(dllexport) void fit_tree(DecisionTree* d_tree, double **arr, int n_rows, int n_cols, bool printTree) { d_tree->fit(arr, n_rows, n_cols, printTree); }
	__declspec(dllexport) size_t* predict(DecisionTree *d_tree, double **arr, int n_rows, int n_cols){ return d_tree->predict_data(arr, n_rows, n_cols); }
}
