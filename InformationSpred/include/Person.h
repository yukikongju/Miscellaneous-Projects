#ifndef PERSON_H
#define PERSON_H

#include <vector>
#include <iostream>
#include <memory>

using namespace std;

class Person {
    protected:

    public:
	int id;
	// vector<unique_ptr<Person>> friends;
	// vector<Person> &friends;
	bool hasInformation;
	Person(int id);

	virtual void printInfos();

	// void addFriend(const Person &friendPerson);
};

#endif
