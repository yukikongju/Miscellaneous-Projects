#include <array>
#include <iostream>
#include <fstream>
#include <string>
#include <tuple>
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

// compute mean and variance given by hashmap TODO
// template <typename Type>
// std::tuple<Count<Type>, Count<Type>> getMeanVariance(Count<Type> countDicts[]) {
//     Count<Type> meanDict, varDict;
//     // int numSimulation = sizeof(*countDicts);

//     // compute mean
    
   
//     // compute std



//     return std::make_tuple(meanDict, varDict);
// }

int main() {
    // init variables
    int numSimulation = 10;
    int initialPopulationCount = 100;
    int maxPopulationCount = 500;
    Count<std::string> typeResults[numSimulation];
    Count<char> sexResults[numSimulation];

    // init percentage dict from json
    HashMap<std::string, double> percentageDict = initDictFromJSON();

    for (int i = 0; i < numSimulation; i++) {
	Population population(percentageDict, initialPopulationCount, maxPopulationCount);
	population.runSimulation();
	population.showStatistics();
	typeResults[i] = population.getTypeCountMap();
	sexResults[i] = population.getSexCountMap();
    }

    // compute mean and variance for all simulations and show
    // Count<std::string> meanType, varType;
    // Count<char> meanSex, varSex;
    // std::tie(meanType, varType) = getMeanVariance(typeResults);
    // std::tie(meanSex, varSex) = getMeanVariance(sexResults);


    return 0;
}
