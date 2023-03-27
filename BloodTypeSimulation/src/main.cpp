#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>

#include "individual.h"
#include "population.h"
#include "HashMap.h"
#include "Count.h"


#include "json.hpp"

using json = nlohmann::json;


HashMap<std::string, double> initDictFromJSON() {
    json j = {
        {"O_positive", 0.3}, 
	{"O_negative", 0.13}, 
	{"A_positive", 0.30}, 
	{"A_negative", 0.08}, 
	{"B_positive", 0.08}, 
	{"B_negative", 0.02}, 
	{"AB_positive", 0.02}, 
	{"AB_negative", 0.01}, 
    };

    // init hashmap from json
    HashMap<std::string, double> percentageDict;
    for (const auto& element: j.items()) {
	percentageDict.put(element.key(), element.value());
    }

    return percentageDict;
}

void testCount() {
    Count<std::string> myCount;

    myCount.add("apple");
    myCount.add("banana");
    myCount.add("apple");
    myCount.add("cherry");
    myCount.add("banana");
    myCount.add("banana");

    myCount.print();
}


int main() {
    // testCount();

    // init percentage dict from json
    HashMap<std::string, double> percentageDict = initDictFromJSON();

    // create population
    Population population(percentageDict, 100, 500);

    // run simulation
    population.runSimulation();

    // show simulation statistics
    population.showStatistics();

    return 0;
}
