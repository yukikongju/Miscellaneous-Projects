#ifndef PERSON_H
#define PERSON_H

#include <vector>
#include <iostream>

using namespace std;

class Person {
    protected:

    public:
	int id;
	vector<Person> friends;
	// vector<Person> &friends;
	bool hasInformation;
	Person(int id);

	virtual void printInfos();

	void addFriend(const Person &friendPerson);
};

#endif
