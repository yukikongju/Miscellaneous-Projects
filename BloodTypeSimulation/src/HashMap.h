#ifndef HASHMAP_H
#define HASHMAP_H

#include <unordered_map>
#include <iostream>

template <typename Key, typename Value>
class HashMap {
    public:
	// Constructor
	HashMap();
	void put(const Key& key, const Value& value);
	bool containsKey(const Key& key) const;
	const Value& get(const Key& key) const;
	void print() const;


    protected:
	std::unordered_map<Key, Value> map;
};

#endif
