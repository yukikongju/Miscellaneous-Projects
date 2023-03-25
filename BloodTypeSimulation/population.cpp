#include <iostream>
#include <string>
#include <unordered_map>

#include "individual.h"
#include "population.h"

typedef std::unordered_map<std::string, double> percentagedict_t;
typedef std::unordered_map<std::string, int> countdict_t;
typedef std::unordered_map<char, int> sexdict_t;

// Constructor
Population::Population(percentagedict_t percentageDict, int maxPopulation) {
    this->percentageDict = percentageDict;
    this->maxPopulation = maxPopulation;
}

// Destructor
Population::~Population() {}


void Population::runSimulation() {
    // init sexCount and typeCount hashmap dict
    sexCount['F'] = 0;
    sexCount['M'] = 0;
    for (auto const pair: percentageDict) {
	typeCount[pair.first] = 0;
    }

    // generate population
    for (int i = 0; i < maxPopulation; i++) {
	// generate individual type
	std::string type = getInitType();

	// generate new individual
	Individual bb(type);

	// increment sexCount and typeCount
	sexCount[bb.getSex()] += 1; 
	typeCount[bb.getType()] += 1;

    }

}

std::string Population::getInitType() {
    // Generate a random number between 0 and 1 (inclusive)
    double randNum = static_cast<double>(std::rand()) / RAND_MAX;

    // Iterate over the key-value pairs in the hashmap
    double runningTotal = 0.0;
    for (auto const pair: percentageDict) {
        runningTotal += pair.second;
        if (runningTotal >= randNum) {
	    return pair.first;
        }
    }
    return "Error";
}


/**********************************************************************
*                         Getter and Setters                         *
**********************************************************************/

sexdict_t Population::getSexCount() {
    return sexCount;
}

countdict_t Population::getTypeCount() {
    return typeCount;
}
