#ifndef HASHMAP_H
#define HASHMAP_H

#include <unordered_map>
#include <iostream>
#include <vector>

template <typename Key, typename Value>
class HashMap {
    public:
	// Constructor
	HashMap();

	// Destructor
	virtual ~HashMap();

	void put(const Key& key, const Value& value);
	void add(const Key& key, const Value& value);
	std::vector<Key> getKeys() const;
	bool containsKey(const Key& key) const;
	const Value& get(const Key& key) const;
	void print() const;

	typename std::unordered_map<Key, Value>::iterator begin() {
	    return map.begin();
	}

	typename std::unordered_map<Key, Value>::iterator end() {
	    return map.end();
	}


    protected:
	std::unordered_map<Key, Value> map;
};

#endif
