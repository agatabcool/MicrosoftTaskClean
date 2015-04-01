#ifndef LOGREGRESSION_H
#define LOGREGRESSION_H

#include "preprocessdata.h"
#include <Eigen/Dense>

struct lr_params{
	double step_size;
	double stop_tol;
	int max_iter;
};

class LogRegression
 {


public:
	LogRegression();
	void train(PreprocessData *pd, lr_params train_params, Eigen::MatrixXd &weights);
	void test(PreprocessData *pd, Eigen::MatrixXd weights, Eigen::MatrixXd &Y_pred);
	void get_test_error(PreprocessData *pd, Eigen::MatrixXd Y_pred, double &error);

private:

	lr_params train_params;
	void getGradient(Eigen::MatrixXd X,Eigen::MatrixXd Y,Eigen::MatrixXd w, Eigen::MatrixXd &grad, double C);
	void appendColumn(Eigen::MatrixXd &mat, double val);
	void print1strow(std::vector<std::vector<double> > &vec2d);

	//maybe this should be in a separate class of functions
	void stdvec2emat(std::vector<std::vector<double> > vec2d, Eigen::MatrixXd &test_mat);
	void stdvec2emat(std::vector<double> vec1d, Eigen::MatrixXd &test_mat);

	void printvec(std::vector<double> Y);
	void printmat(std::vector <std::vector<double> >vec2d);



};

#endif
