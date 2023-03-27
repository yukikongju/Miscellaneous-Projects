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
Population::Population(HashMap<std::string, double> percentageDict, int maxPopulation) {
// Population::Population(std::unordered_map<std::string, double> percentageDict, int maxPopulation) {
    this->percentageDict = percentageDict;
    this->maxPopulation = maxPopulation;
    initSimulation();
}

// Destructor
Population::~Population() {}


void Population::initSimulation() {
    // init sexCount and typeCount hashmap dict
    sexCount.put('F', 0);
    sexCount.put('M', 0);
    // sexCount['F'] = 0;
    // sexCount['M'] = 0;
    for (const auto& pair: percentageDict) {
	// typeCount[pair.first] = 0;
	percentageDict.put(pair.first, pair.second);
    }

    // generate population
    for (int i = 0; i < maxPopulation; i++) {
	// generate individual type
	std::string type = getInitType();

	// generate new individual
	Individual bb(type);

	// increment sexCount and typeCount
	sexCount.add(bb.getSex());
	typeCount.add(bb.getType());

	// TODO: add individuals to female and male list

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

Count<char> Population::getSexCount() {
    return sexCount;
}

Count<std::string> Population::getTypeCount() {
    return typeCount;
}
