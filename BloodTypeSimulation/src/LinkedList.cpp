#include <iostream>
#include <stdexcept>

#include "LinkedList.h"
#include "individual.h"

template <typename Type>
void LinkedList<Type>::push(Type type) {
    Node<Type>* node = new Node<Type> { type, head };
    node->next = this->head;
    this->head = node;
}


template <typename Type>
Type LinkedList<Type>::pop() {
    // check if linked list is empty
    if (this->head == nullptr) {
	throw std::out_of_range("Linked list is empty");
    }

    Node<Type>* node = this->head;
    Type data = node->data;
    this->head = node->next;
    delete node;
    return data;
}


template <typename Type>
Node<Type>* LinkedList<Type>::getHead() {
    return head;
}

// Explicit instantiation for string keys and int values
template class LinkedList<Individual>;

