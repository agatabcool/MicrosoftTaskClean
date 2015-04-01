#ifndef PREPROCESSDATA_H
#define PREPROCESSDATA_H

#include <string>
#include <vector>
#include <numeric>


struct DataStats{
	int num_queries;
	std::vector<std::string> uniq_queries;
	int rev_queries;
	std::vector<std::string> uniq_docs;
	int num_feats;
	std::vector<int> docperquery;
	std::vector<int> rel_per_query;
};

struct ParsedData{
	std::vector<int>relev;
	std::vector<std::string>qid;
	std::vector<std::vector<double> >feats;
	std::vector<std::string>docid;

	DataStats data_stats;
};

class PreprocessData
{

public:
	PreprocessData();
	void loadDataFile(std::string fn);
	ParsedData parsed_data;

private:
	void parse_line(std::string fn);
	void parse_qid(std::string qidstr);
	void parse_docid(std::string qidstr);
	void parse_features(std::vector<std::string> strs);

	void get_stats();

	void printstrs(std::vector<std::string> strs);



};

#endif
