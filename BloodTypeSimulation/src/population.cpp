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
	    females.push(bb);
	} else { // 'M'
	    males.push(bb);
	}

    }

}

// show sexCount and typeCount statistics
void Population::showStatistics() { 
    Count<std::string> typeCount;
    Count<char> sexCount; 

    // add males and females
    Node<Individual>* curr_females = females.getHead();
    while (curr_females != nullptr) {
	typeCount.add(curr_females->data.getType());
	sexCount.add(curr_females->data.getSex());
	curr_females = curr_females->next;
    }
    Node<Individual>* curr_males = males.getHead();
    while (curr_males != nullptr) {
	typeCount.add(curr_males->data.getType());
	sexCount.add(curr_males->data.getSex());
	curr_males = curr_males->next;
    }

    // sex count and perc
    std::cout << "Sex Count and Percentage\n";
    sexCount.print();

    // type count and perc
    std::cout << "\nType Count and Percentage\n";
    typeCount.print();

}

void Population::runSimulation() { // TODO:
    for (int i = 0; i < maxPopulationCount; i++) { 
	Individual mother = females.pop();
	Individual father = males.pop();
	Individual* mother_ptr = &mother;
	Individual* father_ptr = &father;

	// Individual bb(mom, dad);
	Individual bb(mother_ptr, father_ptr);

	// add individuals to female and male list
	if (bb.getSex() == 'F') {
	    females.push(bb);
	} else { // 'M'
	    males.push(bb);
	}

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

