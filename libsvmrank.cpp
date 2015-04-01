#include "libsvmrank.h"
#include <iostream>
#include "crossfilefuncts.h"

Libsvmrank:: Libsvmrank(){
	std::cout << "Libsvmrank constructor!" <<std::endl;
}

void Libsvmrank::init_params(){
	//Parameters
	//svm_parameter param;
	param.svm_type = L2R_RANK;
	param.kernel_type = LINEAR;
	param.degree = 3;
	param.gamma = 0;
	param.coef0 = 0;
//	param.nu = 0.5;
	param.cache_size = 1000;
	param.C = 1;
	param.eps = 1e-3;
//	param.p = 0.1;
//	param.shrinking = 1;
//	param.probability = 0;
//	param.nr_weight = 0;
//	param.weight_label = NULL;
//	param.weight = NULL;
}

void Libsvmrank::librank_test(PreprocessData *pd, svm_model *train_model){
	std::cout << "testing Libsvmrank training" << std::endl;
	double *dvec_t= new double[pd->parsed_data.data_stats.num_queries];
	double *ivec_t= new double[pd->parsed_data.data_stats.num_queries];

	std::vector<double> ivec_d(pd->parsed_data.relev.begin(), pd->parsed_data.relev.end());
	//ivec_t = &ivec_d;
	std::copy(ivec_d.begin(), ivec_d.end(), ivec_t);
	//ivec_t = &ivec_d;

	svm_node* testnode;

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
		svm_node *x_space = new svm_node[num_l+1];
		MakeSparseFeatures (feat_vec, x_space);
		double prob_est;
		//predict_probability(train_model, x_space, prob_est);
		prob_est= svm_predict(train_model, x_space);

		//std::cout <<test_labels[k] << " : " << prob_est[0]<< " : " << prob_est << std::endl;
		std::cout <<test_labels[k] <<  " : " << prob_est << std::endl;
		dvec_t[k]=prob_est;
		k++;
	}

	double *result = new double[2];
	problem prob2;
	int *qvec=new int [pd->parsed_data.data_stats.num_queries];
//	int j_val=0;
//	for(int i = 0; i<15000; i++){
//		if(remainder(i,1000)==0){
//			j_val++;
//		}
//		qvec[i] =j_val;
//	}

	get_queryvec(pd, qvec);
	//get_queryvec(pd, &prob2, qvec);
	int n =pd->parsed_data.data_stats.num_queries;
	svm_eval_list(ivec_t,dvec_t,qvec,n,result);
	std::cout << "Pairwise Accuracy = " << result[0]*100 <<std::endl;
	std::cout << "Mean NDCG = " << result[1] <<std::endl;

}



void Libsvmrank::librank_train(PreprocessData *pd){
	std::cout << "new lr RANK train" <<std::endl;
	init_params();
	svm_model *train_model;
	svm_problem prob;
	svm_check_parameter(&prob,&param);
	get_traindata(pd, &prob);
	std::cout << "starting training" <<std::endl;
	train_model=svm_train(&prob,&param);
//
	const char *model_file_name = "saved_libsvmrank_model.model";
	int saved= svm_save_model(model_file_name, train_model);
	//destroy_param(&param);
}


void Libsvmrank::get_traindata(PreprocessData *pd, svm_problem *prob){
//	struct svm_node
//	{
//		int index;
//		double value;
//	};
//
//	struct svm_problem
//	{
//		int l,n;
//		int *query;
//		double *y;
//		struct svm_node **x;
//	};

	std::cout << "get_traindata" <<std::endl;
	prob->l = pd->parsed_data.data_stats.num_queries;
	prob->n = pd->parsed_data.data_stats.num_feats;

	//Change data struct to Libsvm format
	std::cout << "get_traindata Libsvm formal" <<std::endl;
	get_yvec(pd, prob);
	get_featurevec(pd, prob);
	//int *qvec =new int [pd->parsed_data.data_stats.num_queries];
	get_queryvec(pd, prob);

//	//initialize the problem
//	feature_node** x = new feature_node *[prob->l];
//
//	FeatureVec features = pd->parsed_data.feats;
//	//std::vector <double> ::iterator col_it;
//	int i =0;
//
//	std::cout << "get_traindata row it" <<std::endl;
//
//	for(FeatureVec::iterator it= features.begin(); it!=features.end(); it++, i++){
//		//std::cout << "i: " << i << std::endl;
//		std::vector<double> feat_vec;
//		for (std::vector<double>::iterator col_it = it->begin(); col_it != it->end(); col_it++) {
//			feat_vec.push_back(*col_it);
//		}
//		std::reverse(feat_vec.begin(),feat_vec.end());
//
//		int num_l=0;
//		GetSparseFeatLength(feat_vec, num_l);
//		feature_node *x_space = new feature_node[num_l+1];
//		MakeSparseFeatures (feat_vec, x_space);
//
//		x[i] = x_space;
//		//delete[] x_space;
//		//feat_vec.clear();
//	}
//
//	prob->x = x;

	std::cout << "get_traindata ENDDDDDDD" <<std::endl;
}

void Libsvmrank::get_yvec(PreprocessData *pd, svm_problem *prob){
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

void Libsvmrank::get_featurevec(PreprocessData *pd, svm_problem *prob){
	svm_node** x = new svm_node *[prob->l];
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
		std::reverse(feat_vec.begin(),feat_vec.end());

		int num_l=0;
		GetSparseFeatLength(feat_vec, num_l);
		svm_node *x_space = new svm_node[num_l+1];
		MakeSparseFeatures (feat_vec, x_space);

		x[i] = x_space;

		feat_vec.clear();
	}

	prob->x = x;
}

void Libsvmrank::get_queryvec(PreprocessData *pd, svm_problem *prob){
	//there is a much better way to do this!
	std::cout << "get_queryvec LOG_RANK" <<std::endl;

	std::vector<std::string>::iterator qid_it;
	std::vector<std::string>::iterator qit;

	int j =1;
	int k =0;

	//qvec=new int [pd->parsed_data.data_stats.num_queries];
	prob->query = new int [prob->l];
	for(qit=pd->parsed_data.data_stats.uniq_queries.begin(); qit!=pd->parsed_data.data_stats.uniq_queries.end(); qit++){
		//std::cout << "in" << std::endl;
		for(qid_it=pd->parsed_data.qid.begin(); qid_it!=pd->parsed_data.qid.end(); qid_it++){
			std::string str_qit = *qit;
			if(str_qit.compare(*qid_it) == 0){
				prob->query[k]=j;
				k++;
			}
		}
		j++;
		//std::cout << k << std::endl;
	}
}

void Libsvmrank::get_queryvec(PreprocessData *pd, int *qvec){
	//there is a much better way to do this!
	std::cout << "get_queryvec2 ..." <<std::endl;

	std::vector<std::string>::iterator qid_it;
	std::vector<std::string>::iterator qit;

	int j =1;
	int k =0;

	//qvec=new int [pd->parsed_data.data_stats.num_queries];
	for(qit=pd->parsed_data.data_stats.uniq_queries.begin(); qit!=pd->parsed_data.data_stats.uniq_queries.end(); qit++){
		//std::cout << "in" << std::endl;
		for(qid_it=pd->parsed_data.qid.begin(); qid_it!=pd->parsed_data.qid.end(); qid_it++){
			std::string str_qit = *qit;
			if(str_qit.compare(*qid_it) == 0){
				qvec[k]=j;
				k++;
			}
		}
		j++;
		//std::cout << k << std::endl;
	}
}

