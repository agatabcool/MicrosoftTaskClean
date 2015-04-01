#include "logregression.h"
#include <iostream>
#include <math.h>

#include "linear.h"
//using namespace Eigen;

double apply_exp(double x)
{
    return std::exp(x);
}

double apply_log(double x)
{
    return std::log(x);
}

double add_scalar(double x)
{
    return x+1;
}

double mat_recip(double x)
{
    return 1/x;
}

double is_inf(double x)
{
	if(isinf(x)){
		return std::numeric_limits<double>::max();
	}

	else{
		return x;
	}
}


LogRegression:: LogRegression(){

}

void LogRegression::test(PreprocessData *pd, Eigen::MatrixXd weights, Eigen::MatrixXd &Y_pred){
	std::cout << "testing logistic regression" << std::endl;
	Eigen::MatrixXd X(pd->parsed_data.feats.size(), pd->parsed_data.feats[0].size());
	appendColumn(X, 1);

	//Compute P(Y|X):
	Eigen::MatrixXd PYX(X.rows(), 1);


	Eigen::VectorXd W_row= weights.row(0);
	for(int i=0; i<X.rows(); i++){
		Eigen::VectorXd X_row= X.row(i);
		double val=X_row.dot(W_row);
		PYX(i,0)=1/(1+exp(-1*val));
		if(PYX(i,0)>=0.5){
			Y_pred(i,0)=1;
		}
		else{
			Y_pred(i,0)=0;
		}
	}

}

void LogRegression::get_test_error(PreprocessData *pd, Eigen::MatrixXd Y_pred, double &error){
	//Y = pd->parsed_data.relev;
	std::vector<double> Y_temp(pd->parsed_data.relev.begin(), pd->parsed_data.relev.end());
	Eigen::MatrixXd Y(Y_temp.size(),1);
	stdvec2emat(Y_temp, Y);
	Eigen::MatrixXd err(Y_temp.size(),1);
	for(int i = 0; i<Y_temp.size(); i++){
		if (Y.row(i)==Y_pred.row(i)){
			err(i,0) = 1;
		}
		else{
			err(i,0) = 0;
		}

	}
	error= err.sum()/err.rows();
}


void LogRegression::train(PreprocessData *pd, lr_params train_params, Eigen::MatrixXd &weights){
	double C = 0.001;


	//print_to(eigen, &std::cout);

	//std::vector<std::vector<double> > feats_bias=pd->parsed_data.feats;
	Eigen::MatrixXd feats_bias(pd->parsed_data.feats.size(), pd->parsed_data.feats[0].size());
	appendColumn(feats_bias, 1);

	//initial weights set to zero
	//Eigen::MatrixXd weights(1,pd->parsed_data.data_stats.num_feats+1);
	weights.setZero();

	//change y=0 to y=-1
	std::vector<double> Y_temp(pd->parsed_data.relev.begin(), pd->parsed_data.relev.end());
	std::replace(Y_temp.begin(), Y_temp.end(), 0, -1);
	Eigen::MatrixXd Y(Y_temp.size(),1);
	stdvec2emat(Y_temp, Y);

	Eigen::MatrixXd obj(train_params.max_iter,1);

	for(int i=0; i<train_params.max_iter;i++){
		if(remainder(i,100)==0){
			std::cout << i <<std::endl;
		}

		//exp_part = exp(-Y.*(X*(w')));
		Eigen::MatrixXd Xw;
		Xw = feats_bias*weights.transpose();
		Eigen::MatrixXd Y_neg;
		Y_neg=Y*-1;
		Xw = Xw.cwiseProduct(Y_neg);
		Xw=Xw.unaryExpr(std::ptr_fun(apply_exp));
		Xw=Xw.unaryExpr(std::ptr_fun(is_inf));
		// obj(i) = -sum(log(1 + exp_part));

		Xw=Xw.unaryExpr(std::ptr_fun(add_scalar));
		//Xw=Xw+1;
		Xw=Xw.unaryExpr(std::ptr_fun(apply_log));
		obj(i,0) = Xw.sum()*-1;

		//get the gradient
		Eigen::MatrixXd grad;
		getGradient(feats_bias,Y,weights, grad,C);

		weights = weights + grad*train_params.step_size;

		//Check if threshold has been reached
		double grad_norm;
		grad_norm=grad.norm()/Y.rows();

		if(grad_norm<train_params.stop_tol){
			std::cout << "yay!!" <<std::endl;
			break;
		}
	}

	std::cout << "numeric limit: " << std::numeric_limits<double>::max() <<std::endl;
	//printvec(Xw);

	std::vector<std::vector<double> > vec2d;
	std::vector<double> vec1d;
	for(int j = 0; j<3; j++){
		for(int i = 0; i<3; i++){
			vec1d.push_back(i*i);
		}
		vec2d.push_back(vec1d);
		vec1d.clear();
	}

	Eigen::MatrixXd test_mat(3, 3);
	stdvec2emat(vec2d, test_mat);
	std::cout << "test_mat: " << std::endl;
	std::cout<< test_mat << std::endl;
	std::cout << "test_mat sum: " << test_mat.colwise().sum() << std::endl;
	appendColumn(test_mat, 1);
	std::cout << "test_mat appended: " << std::endl;
	std::cout<< test_mat << std::endl;
	std::cout << "test_mat squared: " << std::endl;
	test_mat.unaryExpr(std::ptr_fun(apply_exp));
	std::cout<< test_mat.unaryExpr(std::ptr_fun(apply_exp)) << std::endl;
}



void LogRegression::stdvec2emat(std::vector<std::vector<double> > vec2d, Eigen::MatrixXd &test_mat){
	for(int i =0; i<test_mat.rows(); i++){
		for(int j =0; j<test_mat.cols(); j++){
			test_mat(i,j) = vec2d[i][j];
		}
	}
}

void LogRegression::stdvec2emat(std::vector<double> vec1d, Eigen::MatrixXd &test_mat){
	for(int i =0; i<test_mat.rows(); i++){
		for(int j =0; j<test_mat.cols(); j++){
			test_mat(i,j) = vec1d[i];
		}
	}
}

void LogRegression::printmat(std::vector <std::vector<double> >vec2d){
	for(std::vector<std::vector<double> >::iterator row_it =vec2d.begin(); row_it!=vec2d.end(); row_it++){
		for(std::vector<double>::iterator col_it =row_it->begin(); col_it!=row_it->end(); col_it++){
			std::cout<< *col_it << " ";
		}
		std::cout<<"\n";
	}
}

void LogRegression::getGradient(Eigen::MatrixXd X,Eigen::MatrixXd Y,Eigen::MatrixXd w, Eigen::MatrixXd &grad, double C){
	//Compute the Logistic Regression gradient.
	// X is a N x P matrix of N examples with P features each
	// Y is a N x 1 vector of (-1, +1) class labels
	// W is a 1 x P weight vector
	// C is the regularization parameter

//	std::cout << "in get gradient" <<std::endl;

	//yx = bsxfun(@times, Y, X);
	Eigen::MatrixXd YX;
	Eigen::MatrixXd Y_rep(X.rows(), X.cols());
	for(int i=0; i<X.cols(); i++){
		Y_rep.col(i)=Y;
	}
	YX = X.cwiseProduct(Y_rep);
//	std::cout << "YX.rows(): "<<YX.rows() <<std::endl;
//	std::cout << "YX.cols(): "<<YX.cols() <<std::endl;


	//wx = dot(X, repmat(w, size(X, 1), 1), 2);
	Eigen::MatrixXd WX(X.rows(),1);
	Eigen::VectorXd W_row= w.row(0);
	for(int i=0; i<X.rows(); i++){
		Eigen::VectorXd X_row= X.row(i);
		WX(i,0)=X_row.dot(W_row);
	}

//	std::cout << "WX.rows(): "<<WX.rows() <<std::endl;
//	std::cout << "WX.cols(): "<<WX.cols() <<std::endl;


	//grad = sum(bsxfun(@times, 1./(1 + exp(Y .* wx)), yx)) - (C * w);
	//Eigen::MatrixXd Xw;
	//Xw = w * X;
	WX = WX.cwiseProduct(Y);
//	std::cout << "WX.rows(): Y.wx: *"<<WX.rows() <<std::endl;
//	std::cout << "WX.cols(): "<<WX.cols() <<std::endl;
	WX=WX.unaryExpr(std::ptr_fun(apply_exp));
	WX=WX.unaryExpr(std::ptr_fun(add_scalar));
	WX=WX.unaryExpr(std::ptr_fun(mat_recip));
//	std::cout << "got here mat_recip" <<std::endl;
//	std::cout << "WX.rows(): "<<WX.rows() <<std::endl;
//	std::cout << "WX.cols(): "<<WX.cols() <<std::endl;
//	std::cout << "YX.rows(): "<<YX.rows() <<std::endl;
//	std::cout << "YX.cols(): "<<YX.cols() <<std::endl;
//	std::cout << "got here YtX" <<std::endl;
	Eigen::MatrixXd WX_rep(YX.rows(), YX.cols());
	for(int i=0; i<YX.cols(); i++){
		WX_rep.col(i)=WX;
	}
	YX = X.cwiseProduct(WX_rep);
	//YX=YX.cwiseProduct(WX);
//	std::cout << "got here cwiseProduct" <<std::endl;
	Eigen::MatrixXd sum_Xw;
	sum_Xw=YX.colwise().sum();
//	std::cout << "got here sum_Xw" <<std::endl;
	grad = sum_Xw -C*w;
//	std::cout << "got here grad" <<std::endl;
}


void LogRegression::appendColumn(Eigen::MatrixXd &mat, double val){
	Eigen::MatrixXd ones_test(mat.rows(),1);
	ones_test.setOnes();
	ones_test=ones_test*val;

	Eigen::MatrixXd mat_temp(mat.rows(), mat.cols());
	mat_temp =mat.rowwise().reverse();
	mat_temp.conservativeResize(mat_temp.rows(), mat_temp.cols()+1);
	mat_temp.col(mat_temp.cols()-1) = ones_test;
	mat.conservativeResize(mat.rows(), mat.cols()+1);
	mat.swap(mat_temp.rowwise().reverse());
}


void LogRegression::printvec(std::vector<double> Y){
	std::vector<double>::iterator col;
	for (col = Y.begin(); col != Y.end(); col++) {
			std::cout << *col <<std::endl;
	}
}

void LogRegression::print1strow(std::vector<std::vector<double> > &vec2d){
		std::vector< std::vector<double> >::iterator row;
		std::vector<double>::iterator col;
			int i =0;
			for (row = vec2d.begin(); row != vec2d.end(); row++) {
				//std::cout << "new row" <<std::endl;
				std::vector<double> new_row;
			    for (col = row->begin(); col != row->end(); col++) {
			    	if(i<1){
			    		std::cout << *col <<std::endl;
			    	}
			    }
			    i++;
			}
}
