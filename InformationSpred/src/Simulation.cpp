#include <Simulation.h>
#include <vector>

Simulation::Simulation(int N, int initialPeopleWithInfo, double meanFriends, double stdFriends): N(N), initialPeopleWithInfo(initialPeopleWithInfo), meanFriends(meanFriends), stdFriends(stdFriends), population(vector<Person>()) {}

void Simulation::simulate() {}


void Simulation::addToPopulation(Person *friendPerson) {
    population.push_back(unique_ptr<Person>(friendPerson));
}
