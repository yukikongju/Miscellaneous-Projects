#ifndef PERSON_BERNOUILLI_H
#define PERSON_BERNOUILLI_H

#include "Person.h"

class PersonBernouilli : public Person {
    protected: 
	double p; // probability of sharing the information

    public:
	PersonBernouilli(int id, double probability);

};

#endif
