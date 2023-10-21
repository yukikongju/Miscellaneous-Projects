#ifndef PERSON_BERNOUILLI_H
#define PERSON_BERNOUILLI_H

#include "Person.h"

class PersonBernouilli : public Person {
    protected: 

    public:
	double p; // probability of sharing the information
	PersonBernouilli(int id, double probability);

	void printInfos() override;

};

#endif
