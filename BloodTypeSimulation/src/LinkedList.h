#ifndef LINKEDLIST_H
#define LINKEDLIST_H

#include <iostream>
#include <string>

template <typename Type>
struct Node {
    Type data;
    Node<Type>* next;
};


template <typename Type>
class LinkedList {
    private:
	Node<Type>* head;

    public: 
	// Constructor
	LinkedList() : head(nullptr) {}

	// Destructor
	virtual ~LinkedList() {}

	void push(Type type);
	Type pop();

	Node<Type>* getHead();

};


#endif
