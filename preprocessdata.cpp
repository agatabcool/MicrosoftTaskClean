#include "preprocessdata.h"
#include <iostream>
#include <fstream>
#include <boost/algorithm/string.hpp>


PreprocessData :: PreprocessData(){
}


void PreprocessData :: loadDataFile(std::string fn){
	std::ifstream file(fn);
	std::cout << "file loaded" << std::endl;

	std::string str;

	while (std::getline(file, str))
	{
		parse_line(str);
	}

	get_stats();
	std::cout << "file parsed" <<std::endl;

}

void PreprocessData :: parse_line(std::string str){
	std::vector<std::string> strs;
	boost::split(strs,str,boost::is_any_of(" "),boost::token_compress_on);

	//Parse doc-id
	parse_docid(strs.back());
	strs.pop_back();
	strs.pop_back();
	strs.pop_back();

	//Once doc-id taken out, reverse list
	std::reverse(strs.begin(), strs.end());

	//Parse relevance id
	int temp_int =std::stoi( strs.back());
	parsed_data.relev.push_back(temp_int);
	strs.pop_back();

	//Parse qid
	parse_qid(strs.back());
	strs.pop_back();

	//Parse features
	parse_features(strs);

}

void PreprocessData ::printstrs(std::vector<std::string> strs){
    for(std::vector<std::string>::iterator it = strs.begin();it!=strs.end();++it){
        std::cout<<*it<<std::endl;
    }
}

void PreprocessData ::parse_qid(std::string qidstr){
	std::vector<std::string> strs;
	boost::split(strs,qidstr,boost::is_any_of(":"),boost::token_compress_on);
	parsed_data.qid.push_back(strs.back());
}

void PreprocessData ::parse_docid(std::string docidstr){
	std::vector<std::string> strs;
	boost::split(strs,docidstr,boost::is_any_of("="),boost::token_compress_on);
	parsed_data.docid.push_back(strs.back());
}

void PreprocessData ::parse_features(std::vector<std::string> strs){

	std::vector<double> feat_vals;

	for(std::vector<std::string>::iterator it= strs.begin() ; it != strs.end(); ++it){
		std::vector<std::string> featstr;
		boost::split(featstr,*it,boost::is_any_of(":"),boost::token_compress_on);
		double temp_val = std::atof(featstr.back().c_str());
		feat_vals.push_back(temp_val);

	}

	parsed_data.feats.push_back(feat_vals);
}

void PreprocessData::get_stats(){

	//Number of total queries
	parsed_data.data_stats.num_queries = parsed_data.relev.size();

	//Number of total queries marked as relevant
	parsed_data.data_stats.rev_queries = std::accumulate(parsed_data.relev.begin(),parsed_data.relev.end(),0);

	//Number of features per query
	parsed_data.data_stats.num_feats = parsed_data.feats[0].size();

	//Unique documents
	parsed_data.data_stats.uniq_docs = parsed_data.docid;
	parsed_data.data_stats.uniq_docs.erase( std::unique( parsed_data.data_stats.uniq_docs.begin(), parsed_data.data_stats.uniq_docs.end() ), parsed_data.data_stats.uniq_docs.end() );

	//Unique queries
	parsed_data.data_stats.uniq_queries = parsed_data.qid;
	parsed_data.data_stats.uniq_queries.erase( std::unique( parsed_data.data_stats.uniq_queries.begin(), parsed_data.data_stats.uniq_queries.end() ), parsed_data.data_stats.uniq_queries.end() );

	///Documents per query
	for (std::vector<std::string>::iterator it = parsed_data.data_stats.uniq_queries.begin() ; it != parsed_data.data_stats.uniq_queries.end(); ++it){
		int mycount = std::count (parsed_data.qid.begin(), parsed_data.qid.end(), *it);
		parsed_data.data_stats.docperquery.push_back(mycount);
	}


	std::vector<std::string> temp_qid;
	std::vector<int> temp_relev = parsed_data.relev;
	int i=0;
	for (std::vector<int>::iterator it = temp_relev.begin() ; it != temp_relev.end(); ++it){
		if (*it ==1){
			temp_qid.push_back(parsed_data.qid[i]);
		}
		i++;
	}

	for (std::vector<std::string>::iterator it = temp_qid.begin() ; it != temp_qid.end(); ++it){
		int mycount = std::count (temp_qid.begin(), temp_qid.end(), *it);
		parsed_data.data_stats.rel_per_query.push_back(mycount);
	}

	parsed_data.data_stats.rel_per_query.erase( std::unique( parsed_data.data_stats.rel_per_query.begin(), parsed_data.data_stats.rel_per_query.end() ), parsed_data.data_stats.rel_per_query.end() );
}
