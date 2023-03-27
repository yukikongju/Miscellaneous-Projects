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

std::string Individual::initType() { 

    // get individual has rh
    std::string rh_string = ((parentF->hasRH() || parentM->hasRH())) ? "positive" : "negative";

    // get individual blood type
    bool hasBloodTypeA = ((parentF->getBloodType() == "A") || parentM->getBloodType() == "A") ? true: false;
    bool hasBloodTypeB = ((parentF->getBloodType() == "B") || parentM->getBloodType() == "B") ? true: false;
    bool hasBloodTypeAB = ((parentF->getBloodType() == "AB") || parentM->getBloodType() == "AB") ? true: false;


    std::string bloodtype_string = "";
    if ((hasBloodTypeAB) || (hasBloodTypeA & hasBloodTypeB)) {
	bloodtype_string = "AB";
    } else if (hasBloodTypeA) {
	bloodtype_string = "A";
    } else if (hasBloodTypeB) {
	bloodtype_string = "B";
    } else {
	bloodtype_string = "O";
    }

    // get blood type string
    std::string blood_string = bloodtype_string + "_" + rh_string;

    return blood_string;
}

bool Individual::hasRH() const {
    // find substring after "_"
    std::string delimiter = "_";
    size_t pos = type.find(delimiter);
    std::string rh = type.substr(pos + delimiter.length());

    // std::cout << type << " " << rh << "\n";
    if (rh == "positive") {
	return true;
    }
    return false;
}

std::string Individual::getBloodType() const {
    std::string delimiter = "_";
    size_t pos = type.find(delimiter);
    std::string bloodType = type.substr(0, pos);

    return bloodType;
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

