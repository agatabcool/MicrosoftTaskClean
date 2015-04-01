#ifndef CROSSFILEFUNCTS_H
#define CROSSFILEFUNCTS_H

#include "linear.h"
#include "svm.h"
#include <Eigen/Dense>
#include <vector>


typedef std::vector < std::vector<double> > FeatureVec;

void MakeSparseFeatures (std::vector<double> &features, svm_node *x_space);
void MakeSparseFeatures (std::vector<double> &features, feature_node *x_space);
void GetSparseFeatLength(std::vector<double> &features, int &num_l);
void stdvec2emat(std::vector<std::vector<double> > vec2d, Eigen::MatrixXd &test_mat);
void stdvec2emat(std::vector<double> vec1d, Eigen::MatrixXd &test_mat);

#endif
