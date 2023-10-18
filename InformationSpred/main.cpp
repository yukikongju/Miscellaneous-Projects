#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>

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

    public:
	Simulation(int N, int initialPeopleWithInfo, double meanFriends, double stdFriends): N(N), initialPeopleWithInfo(initialPeopleWithInfo), meanFriends(meanFriends), stdFriends(stdFriends) {}

	virtual void simulate();
};

class NaiveSimulation: public Simulation {
    protected:


    public: 
	NaiveSimulation(int N, int initialPeopleWithInfo, double meanFriends, double stdFriends) : Simulation(N, initialPeopleWithInfo, meanFriends, stdFriends) {}

	void simulate() override {

	}
};


int main() {


    return 0;
}

