#include "HashMap.h"

#include <iostream>
#include <unordered_map>
#include <vector>

// Constructor
template <typename Key, typename Value>
HashMap<Key, Value>::HashMap() {}

// Destructor
template <typename Key, typename Value>
HashMap<Key, Value>::~HashMap() {}

// put a key-value pair to the hashmap
template <typename Key, typename Value>
void HashMap<Key, Value>::put(const Key& key, const Value& value) {
    map[key] = value;
}

// add to a key-value pair to the hashmap
template <typename Key, typename Value>
void HashMap<Key, Value>::add(const Key& key, const Value& value) {
    map[key] += value;
}

template <typename Key, typename Value>
std::vector<Key> HashMap<Key, Value>::getKeys() const {
    std::vector<Key> keys;
    for (const auto& pair: map) {
	keys.push_back(pair.first);
    }

    return keys;
}

// get
template <typename Key, typename Value>
const Value& HashMap<Key, Value>::get(const Key& key) const {
    return map.at(key);
}

// containsKey
template <typename Key, typename Value>
bool HashMap<Key, Value>::containsKey(const Key& key) const {
    return map.count(key) > 0;
}


// Print out the contents of the hashmap
template <typename Key, typename Value>
void HashMap<Key, Value>::print() const {
    for (const auto pair : map) {
	std::cout << pair.first << ": " << pair.second << std::endl;
    }
}

// Explicit instantiation for string keys and int values
template class HashMap<std::string, int>;
template class HashMap<std::string, double>;
template class HashMap<char, int>;
template class HashMap<char, double>;

