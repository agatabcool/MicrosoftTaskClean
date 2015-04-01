#ifndef PROCESSDATA_H
#define PROCESSDATA_H

#include "qcustomplot.h"
#include "preprocessdata.h"

struct x_y_data{
	QVector<double> x, y;
};

class ProcessData
{
public:
	ProcessData();
	void PlotRelevance(QVector<double> &x, QVector<double> &y, PreprocessData *pd);
};


#endif
