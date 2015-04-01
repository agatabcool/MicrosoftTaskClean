#ifndef LOGREGRESSION_H
#define LOGREGRESSION_H

#include "preprocessdata.h"

struct lr_params{
	//default values
	double step_size =0.00001;
	double stop_tol = 0.0001;
	int max_iter = 1000;

//defaults.step_size = 1e-5;
//defaults.stop_tol = 1e-4;
//defaults.max_iter = 1000;
};

class LogRegression2
 {


public:
	LogRegression2();
	void train(PreprocessData *pd);

private:

	lr_params train_params;
	void getGradient(std::vector<std::vector<double> > X,std::vector<double> Y,std::vector<double> w, double C);
	void appendColumn(std::vector<std::vector<double> > &vec2d, double val);
	void print1strow(std::vector<std::vector<double> > &vec2d);

	//maybe this should be in a separate class of functions
	void matix_multiply(std::vector<std::vector<double> > X, std::vector<double> Y, std::vector<double> &out_vec);
	void multiply_elementwise(std::vector<double> X, std::vector<double> Y, std::vector<double> &out_vec, double const_val);
	void applyexp(std::vector<double> X, std::vector<double> &exp_X, bool inf_bool);
	void summat(std::vector<std::vector<double> > vec2d, std::vector<double> &out_vec, int row_col);
	void mattranspose(std::vector<std::vector<double> > mat_in, std::vector<std::vector<double> > &mat_out);

	void printvec(std::vector<double> Y);
	void printmat(std::vector <std::vector<double> >vec2d);



};

#endif
