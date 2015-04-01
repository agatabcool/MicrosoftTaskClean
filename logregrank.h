#ifndef LOGREGRANK_H
#define LOGREGRANK_H

//#include "linear.h"
#include "preprocessdata.h"
#include "ranksvm.h"

typedef std::vector < std::vector<double> > FeatureVec;

class LogregRank{
public:
	LogregRank();
	void logregrank_train(PreprocessData *pd);
	void logregrank_test(PreprocessData *pd, model *train_model);
private:
	parameter param;
	void init_params();
	void get_traindata(PreprocessData *pd, problem *prob);

	void get_yvec(PreprocessData *pd, problem *prob);
	void get_featurevec(PreprocessData *pd, problem *prob);
	void get_queryvec(PreprocessData *pd, problem *prob);
	void get_queryvec(PreprocessData *pd, int *qvec);


};

#endif
