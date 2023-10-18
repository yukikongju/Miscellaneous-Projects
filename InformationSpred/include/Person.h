#ifndef PERSON_H
#define PERSON_H

#include <vector>
#include <iostream>
using namespace std;

class Person {
    protected:
	int id;
	vector<Person*> friends;
	bool hasInformation;

    public:
	Person(int id);

	void printInfos();
};

#endif
