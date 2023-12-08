#ifndef SIMULATION_H
#define SIMULATION_H

#include "Person.h"
#include <memory>

class Simulation {
    protected:
	// vector<Person> population;

    public:
	int N; // number of people in the simulation
	int initialPeopleWithInfo;
	double meanFriends, stdFriends; // mean and standard deviation of friends
	// vector<Person*> population;
	// vector<Person> &population;
	vector<unique_ptr<Person>> population;

	Simulation(int N, int initialPeopleWithInfo, double meanFriends, double stdFriends);

	virtual void simulate();

	// void addToPopulation(const Person &friendPerson); 
	void addToPopulation(Person *friendPerson); 

};
#endif
