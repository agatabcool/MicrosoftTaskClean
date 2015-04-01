#ifndef LOGREG_H
#define LOGREG_H

#include "linear.h"
#include "preprocessdata.h"

typedef std::vector < std::vector<double> > FeatureVec;

class LogReg{
public:
	LogReg();
	void logreg_train(PreprocessData *pd);
	void logreg_test(PreprocessData *pd, model *train_model);
private:
	parameter param;
	void init_params();
	void get_traindata(PreprocessData *pd, problem *prob);

	void get_yvec(PreprocessData *pd, problem *prob);
	void get_featurevec(PreprocessData &pd, problem *prob);



};

#endif
