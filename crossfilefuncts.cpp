#include "crossfilefuncts.h"
#include <iostream>



void GetSparseFeatLength(std::vector<double> &features, int &num_l){
	for (int k=0; k<features.size(); k++){
		if(features[k]>0){
			num_l++;
		}
	}
}

void MakeSparseFeatures (std::vector<double> &features, feature_node *x_space){
	int j=0;
	for (int k=0; k<features.size(); k++){
		if(features[k]>0){
			x_space[j].index = j+1;
			x_space[j].value = features[k];
			j++;
		}
	}
	x_space[j].index = -1;
}

void MakeSparseFeatures (std::vector<double> &features, svm_node *x_space){
	int j=0;
	for (int k=0; k<features.size(); k++){
		if(features[k]>0){
			x_space[j].index = j+1;
			x_space[j].value = features[k];
			j++;
		}
	}
	x_space[j].index = -1;
}

void stdvec2emat(std::vector<std::vector<double> > vec2d, Eigen::MatrixXd &test_mat){
	for(int i =0; i<test_mat.rows(); i++){
		for(int j =0; j<test_mat.cols(); j++){
			test_mat(i,j) = vec2d[i][j];
		}
	}
}

void stdvec2emat(std::vector<double> vec1d, Eigen::MatrixXd &test_mat){
	for(int i =0; i<test_mat.rows(); i++){
		for(int j =0; j<test_mat.cols(); j++){
			test_mat(i,j) = vec1d[i];
		}
	}
}
