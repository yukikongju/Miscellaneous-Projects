#include <iostream>
#include <string>

#include "individual.h"

// Constructor
Individual::Individual(std::string type) {
    this->type = type;
    this->sex = Individual::initSex();
    this->age = 1;
    this->timeSinceLastChild = -1;
    this->parentF= nullptr;
    this->parentM= nullptr;

}

Individual::Individual(std::string type, char sex) {
    this->type = type;
    this->sex = sex;
    this->age = 1;
    this->timeSinceLastChild = -1;
    this->parentF= nullptr;
    this->parentM= nullptr;
}

Individual::Individual(Individual* mom, Individual* dad) {
    this->sex = Individual::initSex();
    this->age = 1;
    this->timeSinceLastChild = -1;

    this->parentF=mom;
    this->parentM=dad;

    this->type = initType();
}


// Destructor
Individual::~Individual() {}


// initSex
char Individual::initSex(){
    if (rand() % 2 == 0) {
	return 'F';
    }
    return 'M';
}

std::string Individual::initType() { // TODO:
    return "O_positive";
}

/**********************************************************************
*                         Getter and Setters                         *
**********************************************************************/

//
char Individual::getSex() const {
    return sex;
}

std::string Individual::getType() const {
    return type;
}

