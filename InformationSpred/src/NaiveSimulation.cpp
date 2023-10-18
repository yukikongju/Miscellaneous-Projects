#include <NaiveSimulation.h>
#include <PersonBernouilli.h>

NaiveSimulation::NaiveSimulation(int N, int initialPeopleWithInfo, double meanFriends, double stdFriends, double meanProbability, double stdProbability) :
    Simulation(N, initialPeopleWithInfo, meanFriends, stdFriends), 
    meanProbability(meanProbability), stdProbability(stdProbability),
    distributionFriends(meanFriends, stdFriends),
    distributionProbability(meanProbability, stdProbability) {}

void NaiveSimulation::simulate() {
    random_device rd;
    mt19937 gen(rd());

    // 1. create population
    for (int i=0; i<N; ++i) {
	// compute probability
	double probability = max(static_cast<double>(distributionProbability(generator)), 1.0);

	// add person to population
	PersonBernouilli person(i, probability);
	population.push_back(&person);
    }

    // 2. initialize friends and people with information
    for (int i=0; i<N; ++i) {
	// compute number of friends (cannot be friends with yourself lolz)
	int numFriends = max(static_cast<int>(distributionFriends(generator)), N-1);
	vector<int> friendsIndices(numFriends);
	// friendsIndices.push_back(i); // initialize 


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
