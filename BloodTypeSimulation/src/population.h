#ifndef POPULATION_H
#define POPULATION_H

#include <iostream>
#include <string>
#include <unordered_map>

#include "LinkedList.h"
#include "individual.h"
#include "HashMap.h"
#include "Count.h"
#include <list>



class Population {

private:
    int initialPopulationCount;
    int maxPopulationCount;
    int populationCount;
    Count<std::string> typeCountMap;
    Count<char> sexCountMap;
	
    HashMap<std::string, double> percentageDict;

    LinkedList<Individual> females;
    LinkedList<Individual> males;

    std::string getInitType();
    void initSimulation();
    void addNewborn(Individual* newborn_ptr);

public:
	Population(HashMap<std::string, double> percentageDict, int initialPopulationCount, int maxPopulationCount);
	virtual ~Population();
	void runSimulation();
	void showStatistics();

	// Getter and Setters
	Count<std::string> getTypeCountMap() const;
	Count<char> getSexCountMap() const;

};

#endif /* POPULATION_H */
