#include <iostream>
#include <string>
#include <unordered_map>
#include <ctime>
#include <cstdlib>
#include <random>

#include "individual.h"
#include "population.h"
#include "HashMap.h"
#include "Count.h"


// Constructor
Population::Population(HashMap<std::string, double> percentageDict, int maxPopulationCount,
	int initialPopulationCount) {
    this->percentageDict = percentageDict;
    this->maxPopulationCount = maxPopulationCount;
    this->initialPopulationCount = initialPopulationCount;
    initSimulation();
}

// Destructor
Population::~Population() {}


// initialize population
void Population::initSimulation() {
    for (int i = 0; i < initialPopulationCount; i++) {
	// generate individual type
	std::string type = getInitType();

	// generate new individual
	Individual bb(type);
	Individual* bb_ptr = &bb;

	// add newborn to population
	addNewborn(bb_ptr);

    }

}

void Population::addNewborn(Individual* newborn_ptr) {
    // add to sex and type count map
    typeCountMap.increment(newborn_ptr->getType());
    sexCountMap.increment(newborn_ptr->getSex());

    // add individuals to female and male list
    if (newborn_ptr->getSex() == 'F') {
	females.push(*newborn_ptr);
    } else { // 'M'
	males.push(*newborn_ptr);
    }
}

// show sexCount and typeCount statistics
void Population::showStatistics() { 
    std::cout << "\n---------------------------\n";
    std::cout << "Sex Count and Percentage\n";
    sexCountMap.print();

    std::cout << "\n";
    std::cout << "\nType Count and Percentage\n";
    typeCountMap.print();

}

void Population::runSimulation() { // TODO:
    for (int i = 0; i < maxPopulationCount; i++) { 
	Individual mother = females.pop();
	Individual father = males.pop();
	Individual* mother_ptr = &mother;
	Individual* father_ptr = &father;

	// Individual bb(mom, dad);
	Individual bb(mother_ptr, father_ptr);
	Individual* bb_ptr = &bb;

	// add newborn to population
	addNewborn(bb_ptr);

	// put mother and father back in the list
	females.push(mother);
	males.push(father);
    }

}


// Generate blood type for individual following initial distribution
std::string Population::getInitType() {
    // Generate a random number between 0 and 1 (inclusive)
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    std::string lastKey = "";
    double randNum = dis(gen);

    // Iterate over the key-value pairs in the hashmap
    double runningTotal = 0.0;
    for (auto const pair: percentageDict) {
	runningTotal += pair.second;
	if (runningTotal >= randNum) {
	    return pair.first;
	}
	lastKey = pair.first;
    }
    return lastKey;
}


/**********************************************************************
*                         Getter and Setters                         *
**********************************************************************/

Count<std::string> Population::getTypeCountMap() const {
    return typeCountMap;
}

Count<char> Population::getSexCountMap() const {
    return sexCountMap;
}
