#include “logregression2.h"
#include <iostream>
#include <math.h>
#include <Eigen/Dense>



LogRegression2:: LogRegression2(){

}

void LogRegression2::train(PreprocessData *pd){
	double C = 0.001;

	std::vector<std::vector<double> > feats_bias=pd->parsed_data.feats;
	//add a bias color of 1s
	appendColumn(feats_bias, 1);

	//initial weights set to zero
	std::vector<double> weights(pd->parsed_data.data_stats.num_feats+1, 0.0);

	//change y=0 to y=-1
	std::vector<double> Y(pd->parsed_data.relev.begin(), pd->parsed_data.relev.end());
	std::replace(Y.begin(), Y.end(), 0, -1);
//	std::cout << y.front() << std::endl;
//	std::cout << y.back() << std::endl;

	std::vector<double> Xw;
	matix_multiply(feats_bias, weights, Xw);
	std::vector<double> exp_num;
	multiply_elementwise(Xw, Y, exp_num, -1);
	std::vector<double> exp_part;
	applyexp(exp_num, exp_part, 1);

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

	printmat(vec2d);

	std::vector<double> vec2d_sum;
	summat(vec2d, vec2d_sum, 2);
	printvec(vec2d_sum);

	for(int i =0; i<train_params.max_iter; i++){
		//exp_part = exp(-Y.*(X*(w')));


	}
	//print1strow(feats_bias);
}

void LogRegression2::printmat(std::vector <std::vector<double> >vec2d){
	for(std::vector<std::vector<double> >::iterator row_it =vec2d.begin(); row_it!=vec2d.end(); row_it++){
		for(std::vector<double>::iterator col_it =row_it->begin(); col_it!=row_it->end(); col_it++){
			std::cout<< *col_it << " ";
		}
		std::cout<<"\n";
	}
}

void LogRegression2::getGradient(std::vector<std::vector<double> > X,std::vector<double> Y,std::vector<double> w, double C){
	//Compute the Logistic Regression gradient.
	// X is a N x P matrix of N examples with P features each
	// Y is a N x 1 vector of (-1, +1) class labels
	// W is a 1 x P weight vector
	// C is the regularization parameter

	//yx = bsxfun(@times, Y, X);
	//X is now YX
	std::vector<std::vector<double> > YX;
	std::vector<std::vector<double> >::iterator row_it;
	std::vector<double >::iterator y_it;
	int i=0;
	for(row_it=X.begin(); row_it!=X.end(); row_it++){
		std::transform(row_it->begin(), row_it->end(), row_it->begin(),
				std::bind1st(std::multiplies<double>(),Y[i]));
		i++;
	}

	//wx = dot(X, repmat(w, size(X, 1), 1), 2);

	std::vector<double> wx;
	for(row_it=X.begin(); row_it!=X.end(); row_it++){
		wx.push_back(std::inner_product(row_it->begin(),row_it->end(),Y.begin(),0.0));
	}
	std::reverse(wx.begin(),wx.end());

	//grad = sum(bsxfun(@times, 1./(1 + exp(Y .* wx)), yx)) - (C * w);
	std::vector<double> denom1;
	multiply_elementwise(Y, wx, denom1, 1);
	std::vector<double> exp_denom1;
	applyexp(denom1, exp_denom1, 0);
	std::vector<double> exp_denom2;
	for(std::vector<double>::iterator it=exp_denom1.begin(); it!=exp_denom1.end();it++){
		double tp = *it;
		exp_denom2.push_back(1/tp);
	}

	std::vector<double> first_ele;
	matix_multiply(X, exp_denom2, first_ele);


}

void LogRegression2::applyexp(std::vector<double> X, std::vector<double> &exp_X, bool inf_bool){
	std::vector<double>::iterator it;
	for (it=X.begin(); it!=X.end(); it++){
		double temp_val = exp(*it);
		if(inf_bool){
			if (isinf(temp_val)){
				exp_X.push_back(std::numeric_limits<double>::max());
			}
			else
			{
				exp_X.push_back(temp_val);
			}
		}
		else{
			exp_X.push_back(exp(*it));
		}

	}

	std::reverse(exp_X.begin(),exp_X.end());
}

void LogRegression2::summat(std::vector<std::vector<double> > vec2d, std::vector<double> &out_vec, int row_col){

	if(row_col==2){
		std::vector<std::vector<double> > vec2d_t;
		mattranspose(vec2d, vec2d_t);
		std::vector<std::vector<double> >::iterator row_it;
		for (row_it = vec2d_t.begin(); row_it != vec2d_t.end(); row_it++) {
			out_vec.push_back(std::accumulate( row_it->begin(), row_it->end(), 0.0 ));
		}
	}

	else{
		std::vector<std::vector<double> >::iterator row_it;
		for (row_it = vec2d.begin(); row_it != vec2d.end(); row_it++) {
			out_vec.push_back(std::accumulate( row_it->begin(), row_it->end(), 0.0 ));
		}
	}

	std::reverse(out_vec.begin(), out_vec.end());
}

void LogRegression2::mattranspose(std::vector<std::vector<double> > mat_in, std::vector<std::vector<double> > &mat_out){
	std::cout << "mattranspose" <<std::endl;
	double mat_temp[mat_in.size()][mat_in[0].size()];
	for (size_t i = 0; i < mat_in.size(); ++i){
		for (size_t j = 0; j < mat_in[0].size(); ++j){
			std::cout << "mattransposing "<< mat_in[i][j] <<std::endl;
			mat_temp[j][i] = mat_in[i][j];
		}
	}
	//mat_out=mat_temp;
}

void LogRegression2::appendColumn(std::vector<std::vector<double> > &vec2d, double val){
	//is there a better way to do this?
	std::vector<std::vector<double> > vec_new;
	std::vector< std::vector<double> >::iterator row;
	std::vector<double>::iterator col;
	for (row = vec2d.begin(); row != vec2d.end(); row++) {
		std::vector<double> new_row;
	    for (col = row->begin(); col != row->end(); col++) {
	    	new_row.push_back(*col);
	    }
	    new_row.push_back(val);
	    std::reverse(new_row.begin(),new_row.end());
	    vec_new.push_back(new_row);
	}

	vec2d.clear();
	vec2d=vec_new;
}

void LogRegression2::matix_multiply(std::vector<std::vector<double> > X, std::vector<double> Y, std::vector<double> &out_vec){

	int X_rows = X.size();
	int X_cols = X[0].size();
	int Y_rows = Y.size();

	std::vector<double>::iterator it;
	for(int i = 0; i<X_rows; i++){
		double out_val=0;
		for(int j=0; j<Y_rows; j++){
			//std::cout << i << " " << j << " " << X[i][j] << " " << Y[j] << std::endl;
			out_val+=X[i][j]*Y[j];
		}

		//std::cout << out_val << std::endl;
		it = out_vec.begin();
		it = out_vec.insert ( it , out_val );
		//out_vec[i] = out_val;
		//std::cout << "out_val again" << std::endl;
	}

}

void LogRegression2::multiply_elementwise(std::vector<double> X, std::vector<double> Y, std::vector<double> &out_vec, double const_val){

	if(X.size() == Y.size()){
		std::transform(Y.begin(), Y.end(), Y.begin(),
		               std::bind1st(std::multiplies<double>(),const_val));

		for(int i =0; i<X.size(); i++){
			out_vec.push_back(X[i]*Y[i]);
		}
		std::reverse(out_vec.begin(), out_vec.end());
	}


	else{
		std::cout << "Vectors X and Y must be of same length." << std::endl;
	}

}

void LogRegression2::printvec(std::vector<double> Y){
	std::vector<double>::iterator col;
	for (col = Y.begin(); col != Y.end(); col++) {
			std::cout << *col <<std::endl;
	}
}

void LogRegression2::print1strow(std::vector<std::vector<double> > &vec2d){
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
