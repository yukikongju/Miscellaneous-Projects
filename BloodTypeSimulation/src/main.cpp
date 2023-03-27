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


HashMap<std::string, double> init_dict_from_json() {
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


int main() {
    // init percentage dict from json
    HashMap<std::string, double> percentageDict = init_dict_from_json();

    // create population
    Population population(percentageDict, 500);
    population.initSimulation();

    std::cout << "Sex Count and Percentage\n";
    population.getSexCount().print();
    population.getSexCount().computePercentages().print();

    std::cout << "\n Type Count and Percentage\n";
    population.getTypeCount().print();
    population.getTypeCount().computePercentages().print();

    return 0;
}
