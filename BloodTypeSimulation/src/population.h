#ifndef POPULATION_H
#define POPULATION_H

#include <iostream>
#include <string>
#include <unordered_map>

#include "individual.h"

typedef std::unordered_map<std::string, double> percentagedict_t;
typedef std::unordered_map<std::string, int> countdict_t;
typedef std::unordered_map<char, int> sexdict_t;

class Population {

private:
    percentagedict_t percentageDict;
    int maxPopulation;
    int populationCount;
    countdict_t typeCount;
    sexdict_t sexCount; // {'F': 50, 'M':52}
	
    std::string getInitType();
	

public:
	Population(std::unordered_map<std::string, double> percentageDict, 
		int maxPopulation);
	~Population();
	void initSimulation();
	sexdict_t getSexCount();
	countdict_t getTypeCount();

};

#endif /* POPULATION_H */
