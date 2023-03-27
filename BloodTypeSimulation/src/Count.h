#ifndef COUNT_H
#define COUNT_H

#include <unordered_map>
#include <iostream>

#include "HashMap.h"


template <typename Key>
class Count : public HashMap<Key, int>{
    public:
	Count();
	void add(const Key& key);
	HashMap<Key, double> computePercentages() const;
	void print() const;

};

#endif

