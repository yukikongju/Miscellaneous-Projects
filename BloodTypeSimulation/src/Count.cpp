#include "Count.h"
#include "HashMap.h"

#include <iostream>
#include <unordered_map>

// Constructor
template <typename Key>
Count<Key>::Count() {}


// Add a key-value pair to the hashmap
template <typename Key>
void Count<Key>::add(const Key& key) {
    this->map[key]++;
}

template <typename Key>
HashMap<Key, double> Count<Key>::computePercentages() const {
    HashMap<Key, double> percentageMap;

    // get max count
    int totalCount = 0;
    for (const auto& pair: this->map) {
	totalCount += pair.second;
    }

    for (const auto& pair: this->map) {
	double percentage = pair.second / (double) totalCount;
	percentageMap.put(pair.first, percentage);
    }

    return percentageMap;
}


// Explicit instantiation for string keys and int values
template class Count<std::string>;
template class Count<char>;

