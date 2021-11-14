#include "structures.h"
#include <algorithm>


vector<double>* sort_column(vector<vector<double>> data, size_t idx);
MajorLabel get_major_label(vector<vector<double>> data, size_t lbl);
size_t dfs(Node node, vector<double> row, int idx);
void dfs_print(Node root, int depth);