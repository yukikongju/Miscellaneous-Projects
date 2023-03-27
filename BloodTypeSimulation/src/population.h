#ifndef POPULATION_H
#define POPULATION_H

#include <iostream>
#include <string>
#include <unordered_map>

#include "individual.h"
#include "HashMap.h"
#include "Count.h"
#include <list>



class Population {

private:
    int initialPopulationCount;
    int maxPopulationCount;
    int populationCount;
	
    HashMap<std::string, double> percentageDict;
    std::list<Individual> females;
    std::list<Individual> males;

    std::string getInitType();
    void initSimulation();

public:
	Population(HashMap<std::string, double> percentageDict, int initialPopulationCount, int maxPopulationCount);
	~Population();
	void runSimulation();
	void showStatistics();
	

};

#endif /* POPULATION_H */
