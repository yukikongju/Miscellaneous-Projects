#include <iostream>
#include <string>

#include "individual.h"

// Constructor
Individual::Individual(std::string type) {
	this->type = type;
	this->sex = Individual::initSex();
	this->age = 1;
	this->timeSinceLastChild = -1;
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

