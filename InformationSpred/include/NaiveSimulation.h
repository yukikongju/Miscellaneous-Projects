#ifndef NAIVE_SIMULATION_H
#define NAIVE_SIMULATION_H

#include "Simulation.h"
#include <algorithm>
#include <chrono>
#include <vector>
#include <random>

class NaiveSimulation: public Simulation {
    protected:
	double meanProbability, stdProbability;
	normal_distribution<double> distributionFriends;
	normal_distribution<double> distributionProbability;

	default_random_engine generator;


    public: 
	NaiveSimulation(int N, int initialPeopleWithInfo, double meanFriends, double stdFriends, double meanProbability, double stdProbability);

	void simulate() override;
};


#endif

