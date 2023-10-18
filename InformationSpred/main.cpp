#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <algorithm>

using namespace std;

class Person {
    protected:
	int id;
	vector<Person*> friends;
	bool hasInformation;

    public:
	Person(int id): id(id), hasInformation(false) {}

};

class PersonBernouilli : public Person {
    protected: 
	double p; // probability of sharing the information

    public:
	PersonBernouilli(int id, double probability) : Person(id), p(probability) {}

};


class Simulation {
    protected:
	int N; // number of people in the simulation
	int initialPeopleWithInfo;
	double meanFriends, stdFriends; // mean and standard deviation of friends
	vector<Person> population;

    public:
	Simulation(int N, int initialPeopleWithInfo, double meanFriends, double stdFriends): N(N), initialPeopleWithInfo(initialPeopleWithInfo), meanFriends(meanFriends), stdFriends(stdFriends), population(N) {}

	virtual void simulate();
};

class NaiveSimulation: public Simulation {
    protected:
	double meanProbability, stdProbability;
	normal_distribution<double> distributionFriends;
	normal_distribution<double> distributionProbability;

	default_random_engine generator;


    public: 
	NaiveSimulation(int N, int initialPeopleWithInfo, double meanFriends, double stdFriends, double meanProbability, double stdProbability) :
	    Simulation(N, initialPeopleWithInfo, meanFriends, stdFriends), 
	    meanProbability(meanProbability), stdProbability(stdProbability),
	    distributionFriends(meanFriends, stdFriends),
	    distributionProbability(meanProbability, stdProbability) {}


	void simulate() override {
	    random_device rd;
	    mt19937 gen(rd());

	    // 1. create population
	    for (int i=0; i<N; ++i) {
		// compute probability
		double probability = max(static_cast<double>(distributionProbability(generator)), 1.0);

		// add person to population
		PersonBernouilli person(i, probability);
		population.push_back(person);
	    }

	    // 2. initialize friends and people with information
	    for (int i=0; i<N; ++i) {
		// compute number of friends (cannot be friends with yourself lolz)
		int numFriends = max(static_cast<int>(distributionFriends(generator)), N-1);
		vector<int> friendsIndices(numFriends);
		while(friendsIndices.size() < numFriends) {
		    int randomInt = uniform_int_distribution<int>(0, N-1)(gen);

		    if (find(friendsIndices.begin(), friendsIndices.end(), randomInt) == friendsIndices.end()) {
			friendsIndices.push_back(randomInt);
		    }
		}


		// TODO: set probability and friends

	    }


	    // 3. run simulation until everyone has information

	}
};


int main() {
    // set initial parameters
    int N = 10, initialPeopleWithInfo = 1;
    double meanFriends = 3.0, stdFriends = 2.0;
    double meanProbability = 0.2, stdProbability = 0.05;

    // run the simulations
    int numSimulations = 100;
    vector<pair<double, double>> results; // <mean, std>

    // TODO: compute monte carlo
    NaiveSimulation simulation(N, initialPeopleWithInfo, meanFriends, 
	    stdFriends, meanProbability, stdProbability);
    simulation.simulate();

    return 0;
}

