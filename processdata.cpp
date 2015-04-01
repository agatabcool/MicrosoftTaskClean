#include "processdata.h"
#include <iostream>
#include <fstream>
#include <numeric>

ProcessData :: ProcessData(){
	std::cout << "Constuctor for processdata" <<std::endl;
}

void ProcessData :: PlotRelevance(QVector<double> &x, QVector<double> &y, PreprocessData *pd){
	std::cout <<"plotting relevance" <<std::endl;
	std::cout << pd->parsed_data.relev.size() << std::endl;
	int sum_of_elems=0;
	sum_of_elems=std::accumulate(pd->parsed_data.relev.begin(),pd->parsed_data.relev.end(),0);
	std::cout << "Sume of relevance " << sum_of_elems << std::endl;

	std::vector<double> v_double(pd->parsed_data.relev.begin(), pd->parsed_data.relev.end());
	y = QVector<double>::fromStdVector(v_double);
	std::cout <<"plotting relevance 2" <<std::endl;
	int sz_rel =pd->parsed_data.relev.size();

	std::vector<double> temp_vec;
	for (int i=0; i<sz_rel; ++i)
	{
		temp_vec[i]; // x goes from -1 to 1
	}

	x = QVector<double>::fromStdVector(temp_vec);
}
