#include "logreg.h"
#include <Eigen/Dense>
#include <iostream>
#include "ranksvm.h"
#include "crossfilefuncts.h"

LogReg::LogReg(){

}

void LogReg::init_params(){
	param.solver_type=L2R_LR;
	param.eps= 0.0001;
	param.C=0.001;
	param.nr_weight=0;
	double w_val=1;
	param.weight=&w_val;
	param.p=0.00001;
}

void LogReg::logreg_test(PreprocessData *pd, model *train_model){
	std::cout << "testing training" << std::endl;
	feature_node* testnode;

	//Change data struct to Libsvm format
	std::vector<int>test_labels, pred_labels;
	test_labels=pd->parsed_data.relev;

	FeatureVec features = pd->parsed_data.feats;
	std::vector <double> ::iterator col_it;
	int k =0;
	for(std::vector<std::vector <double> >::iterator it= features.begin(); it!=features.end(); it++){
		std::vector<double> feat_vec;
		for (col_it = it->begin(); col_it != it->end(); col_it++) {
			feat_vec.push_back(*col_it);
		}
		std::reverse(feat_vec.begin(),feat_vec.end());

		int num_l=0;
		GetSparseFeatLength(feat_vec, num_l);
		feature_node *x_space = new feature_node[num_l+1];
		MakeSparseFeatures (feat_vec, x_space);
		double prob_est[2];
		predict_probability(train_model, x_space, prob_est);
		//prob_est= predict(train_model, x_space);

		//std::cout <<test_labels[k] << " : " << prob_est[0] << std::endl;
		k++;
//		if (prob_est[1]==1){
//
//		}
	}


}



void LogReg::logreg_train(PreprocessData *pd){
	std::cout << "new lr train" <<std::endl;
	init_params();
	model *train_model;
	problem prob;
	check_parameter(&prob,&param);
	get_traindata(pd, &prob);
	train_model=train(&prob,&param);

	const char *model_file_name = "saved_model.model";
	int saved= save_model(model_file_name, train_model);
	//destroy_param(&param);
}

void LogReg::get_traindata(PreprocessData *pd, problem *prob){
//	struct problem
//	{
//		int l, n;
//		int *y;
//		struct feature_node **x;
//		double bias;
//	};

	std::cout << "get_traindata" <<std::endl;
	prob->l = pd->parsed_data.data_stats.num_queries;
	prob->n = pd->parsed_data.data_stats.num_feats;
	prob->bias=0.0;

	//initialize the problem
	feature_node** x = new feature_node *[prob->l];

	get_yvec(pd, prob);
//	get_featurevec(*pd, prob);

	FeatureVec features = pd->parsed_data.feats;
	//std::vector <double> ::iterator col_it;
	int i =0;

	std::cout << "get_traindata row it" <<std::endl;

	for(FeatureVec::iterator it= features.begin(); it!=features.end(); it++, i++){
		std::vector<double> feat_vec;
		for (std::vector <double> ::iterator col_it = it->begin(); col_it != it->end(); col_it++) {
			//std::cout << *col_it << std::endl;
			feat_vec.push_back(*col_it);

		}

		//std::cout << feat_vec.size() << std::endl;
		std::reverse(feat_vec.begin(),feat_vec.end());

		int num_l=0;
		GetSparseFeatLength(feat_vec, num_l);
		feature_node *x_space = new feature_node[num_l+1];
//		std::cout << num_l+1 << std::endl;
		MakeSparseFeatures (feat_vec, x_space);

		x[i] = x_space;

		//feat_vec.clear();
	}

	prob->x = x;
	std::cout << "get_traindata ENDDDDDDD" <<std::endl;
}

//void LogReg::get_featurevec(PreprocessData &pd, problem *prob){
//	feature_node** x = new feature_node *[prob->l];
//	FeatureVec features = pd.parsed_data.feats;
//	//std::vector <double> ::iterator col_it;
//	int i =0;
//
//	std::cout << "get_traindata row it" <<std::endl;
//
//	for(FeatureVec::iterator it= features.begin(); it!=features.end(); it++, i++){
//		std::vector<double> feat_vec;
//		for (std::vector <double> ::iterator col_it = it->begin(); col_it != it->end(); col_it++) {
//			//std::cout << *col_it << std::endl;
//			feat_vec.push_back(*col_it);
//
//		}
//
//		//std::cout << feat_vec.size() << std::endl;
//		std::reverse(feat_vec.begin(),feat_vec.end());
//
//		int num_l=0;
//		GetSparseFeatLength(feat_vec, num_l);
//		feature_node *x_space = new feature_node[num_l+1];
//		//		std::cout << num_l+1 << std::endl;
//		MakeSparseFeatures (feat_vec, x_space);
//
//		x[i] = x_space;
//
//		//feat_vec.clear();
//	}
//
//	prob->x = x;
//}

void LogReg::get_yvec(PreprocessData *pd, problem *prob){
	std::cout << "get_traindata ys" <<std::endl;
	//prob->y = new double(prob->l);
	double y_test[prob->l];
	std::cout << "get_traindata y init " << prob->l<<std::endl;
	for(int i =0; i<prob->l; i++){
		//std::cout <<i<<std::endl;
		y_test[i] = pd->parsed_data.relev[i];
	}
	prob->y=y_test;
}
