#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QApplication>
#include <QTextEdit>
#include <QPushButton>
#include "qcustomplot.h"
#include "preprocessdata.h"

//struct DataStat{
//	int num_queries;
//	int uniq_queries;
//	int rev_queries;
//	int uniq_docs;
//	int num_feats;
//	std::vector<int> docperquery;
//};

 class MainWindow :public QWidget
 {
     Q_OBJECT

public:
     MainWindow(QWidget *parent = 0);
//public slots:
	QCustomPlot * customPlot;
	PreprocessData *pd;
	QCPBars *relevPd_plot;
	//DataStat ds;


private slots:
    // void open();
	void loadFile();
	//void plotRelev(QCustomPlot *customPlot);
	void plotRelev();
	void apply_lr();
	void apply_lr_rank();
	void apply_libsvmrank();


private:
	QPushButton *loadButton;
	QPushButton *quitButton;
	QPushButton *plotRelevButton;
	QPushButton *applyLogRegress;
	QPushButton *applyLogRegress_rank;
	QPushButton *applyLibsvm_rank;
	QTextEdit *textEdit;

	void printSummary();
	void plotRelevPD();

};

#endif
