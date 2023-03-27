#include <array>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <math.h>
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

// template <typename Key, typename Value> 
template <typename Key>
std::tuple<HashMap<Key, double>, HashMap<Key, double>> computeMeanVariance(Count<Key> countDicts[], int numSimulation) {
    HashMap<Key, double> meanDict, varDict;

    // compute mean: $\bar{x} = \frac{1}{n} \sum{x_i}$
    for (int i = 0; i < numSimulation; i++) {
	for (const auto&[key, value] : countDicts[i].computePercentages()) {
	    meanDict.add(key, value);
	}
    }
    for (const auto& [key, value] : meanDict) {
	double mean = value / (double) numSimulation;
	meanDict.put(key, mean);
    }
   
    // compute variance: $var(x) = \frac{1}{n-1} \sum{(x_i - \bar{x})^2}$
    for (int i = 0; i < numSimulation; i++) {
    	for (const auto& [key, value] : countDicts[i].computePercentages()) {
	    varDict.add(key, pow(value - meanDict.get(key), 2));
    	}
    }
    for (const auto& [key, value] : varDict) {
	double var = value / (double) (numSimulation - 1);
	varDict.put(key, var);
    }

    return std::make_tuple(meanDict, varDict);
}

template <typename Key, typename Value>
void printSimulationResults(HashMap<Key, Value> meanDict, HashMap<Key, Value> varDict) {

    auto keys = meanDict.getKeys();
    for (auto key: keys) {
	double mean = meanDict.get(key);
	double var = varDict.get(key);
	std::cout << std::setprecision(3) << key << " => " << " [ Mean : " << mean << " ] ;" << " [ Variance: " << var << " ]\n";
    }

}

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
	// population.showStatistics();
	typeResults[i] = population.getTypeCountMap();
	sexResults[i] = population.getSexCountMap();
    }

    // compute mean and variance for all simulations and show
    auto [meanType, varType] = computeMeanVariance(typeResults, numSimulation);
    auto [meanSex, varSex] = computeMeanVariance(sexResults, numSimulation);

    std::cout << "\nBlood Type Simulation Results\n";
    printSimulationResults(meanType, varType);
    std::cout << "\nSex Simulation Results\n";
    printSimulationResults(meanSex, varSex);

    return 0;
}
