#ifndef LIBSVMRANK_H
#define LIBSVMRANK_H

#include "preprocessdata.h"
#include "svm.h"

typedef std::vector < std::vector<double> > FeatureVec;

class Libsvmrank{
public:
	Libsvmrank();
	void librank_train(PreprocessData *pd);
	void librank_test(PreprocessData *pd, svm_model *train_model);
private:
	svm_parameter param;
	void init_params();
	void get_traindata(PreprocessData *pd, svm_problem *prob);

	void get_yvec(PreprocessData *pd, svm_problem *prob);
	void get_featurevec(PreprocessData *pd, svm_problem *prob);
	void get_queryvec(PreprocessData *pd, svm_problem *prob);
	void get_queryvec(PreprocessData *pd, int *qvec);


};

#endif
