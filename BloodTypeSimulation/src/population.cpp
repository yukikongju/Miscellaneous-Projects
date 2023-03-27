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

	// add individuals to female and male list
	if (bb.getSex() == 'F') {
	    females.push_back(bb);
	} else { // 'M'
	    males.push_back(bb);
	}

    }

}

// show sexCount and typeCount statistics
void Population::showStatistics() { 
    Count<std::string> typeCount;
    Count<char> sexCount; 

    // add males and females
    for (auto ind = females.begin(); ind != females.end(); ++ind) {
	typeCount.add(ind->getType());
	sexCount.add(ind->getSex());
    }
    for (auto ind = males.begin(); ind != males.end(); ++ind) {
	typeCount.add(ind->getType());
	sexCount.add(ind->getSex());
    }

    // sex count and perc
    std::cout << "Sex Count and Percentage\n";
    sexCount.print();
    sexCount.computePercentages().print();

    // type count and perc
    std::cout << "\nType Count and Percentage\n";
    typeCount.print();
    typeCount.computePercentages().print();

}

void Population::runSimulation() { // TODO:
    Individual* mom = new Individual("O_positive", 'F');
    Individual* dad=new Individual("B_negative", 'M');
    for (int i = 0; i < maxPopulationCount; i++) {
	Individual bb(mom, dad);
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

