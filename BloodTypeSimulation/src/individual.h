#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

#include <iostream>
#include <string>


class Individual {
private:
	char sex;
	std::string type;
	int age;
	double wantsToProcreate; // between 0 and 1
	int timeSinceLastChild;
	Individual* parentF; // pointer to female parent
	Individual* parentM; // pointer to male parent
	
	char initSex();
	std::string initType();

public:

	// Constructor and Destructor
	Individual(std::string type);
	Individual(std::string type, char sex);
	Individual(Individual* parentF, Individual* parentM);
	~Individual();

	bool hasRH() const;
	std::string getBloodType() const;
	char getSex() const;
	std::string getType() const;

};

#endif /* INDIVIDUAL_H */
