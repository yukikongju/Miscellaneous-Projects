#ifndef SIMULATION_H
#define SIMULATION_H

#include "Person.h"

class Simulation {
    protected:
	int N; // number of people in the simulation
	int initialPeopleWithInfo;
	double meanFriends, stdFriends; // mean and standard deviation of friends
	vector<Person*> population;

    public:
	Simulation(int N, int initialPeopleWithInfo, double meanFriends, double stdFriends);

	virtual void simulate();

};
#endif
