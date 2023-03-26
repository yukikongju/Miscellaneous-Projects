#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>

#include "individual.h"
#include "population.h"

#include "json.hpp"


typedef std::unordered_map<std::string, double> percentagedict_t;
typedef std::unordered_map<std::string, int> countdict_t;

using json = nlohmann::json;


percentagedict_t init_dict_from_json() {
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
    percentagedict_t percentage_dict;
    for (auto& element: j.items()) {
	percentage_dict[element.key()] = (double) element.value();
    }

    return percentage_dict;
}

void printPercentageDict(percentagedict_t percentage_dict) {
    for (auto const& pair: percentage_dict) {
	std::cout << pair.first << ": " << pair.second << "\n";
    }
}

void printCountDict(countdict_t countDict) {
    for (auto const& pair: countDict) {
	std::cout << pair.first << ": " << pair.second << "\n";
    }
}

void printSexDict(sexdict_t countDict) {
    for (auto const& pair: countDict) {
	std::cout << pair.first << ": " << pair.second << "\n";
    }
}

int main() {
    // init percentage dict from json
    std::unordered_map<std::string, double> percentage_dict = init_dict_from_json();

    // create population
    Population population(percentage_dict, 100);
    population.runSimulation();

    // get sexCount and typeCount
    printCountDict(population.getTypeCount());
    printSexDict(population.getSexCount());


    return 0;
}
