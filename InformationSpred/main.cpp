#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <algorithm>

using namespace std;

#include "NaiveSimulation.h"

int main() {
    // set initial parameters
    int N = 10, initialPeopleWithInfo = 1;
    double meanFriends = 3.0, stdFriends = 2.0;
    double meanProbability = 0.2, stdProbability = 0.05;

    // run the simulations
    int numSimulations = 100;
    vector<pair<double, double>> results; // <mean, std>

    // TODO: compute monte carlo + store them in file
    NaiveSimulation simulation(N, initialPeopleWithInfo, meanFriends, 
	    stdFriends, meanProbability, stdProbability);
    simulation.simulate();

    //

    return 0;
}

