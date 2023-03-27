#ifndef COUNT_H
#define COUNT_H

#include <unordered_map>
#include <iostream>

#include "HashMap.h"


template <typename Key>
class Count : public HashMap<Key, int>{
    public:
	// Constructor
	Count();

	// Destructor
	virtual ~Count();

	void increment(const Key& key);
	HashMap<Key, double> computePercentages() const;
	void print() const;

};

#endif

