#ifndef POPULATION_H
#define POPULATION_H

#include <iostream>
#include <string>
#include <unordered_map>

#include "individual.h"
#include "HashMap.h"
#include "Count.h"


class Population {

private:
    int maxPopulation;
    int populationCount;
	
    HashMap<std::string, double> percentageDict;
    Count<std::string> typeCount;
    Count<char> sexCount; // {'F': 50, 'M':52}


    std::string getInitType();
	

public:
	Population(HashMap<std::string, double> percentageDict, int maxPopulation);
	~Population();
	void initSimulation();
	
	Count<std::string> getTypeCount();
	Count<char> getSexCount();

};

#endif /* POPULATION_H */
