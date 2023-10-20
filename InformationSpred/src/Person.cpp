#include <Person.h>

Person::Person(int id): id(id), hasInformation(false) {}

void Person::printInfos() {
    cout << "id: " << id << "; hasInformation: " << hasInformation << "; number of friends: " << friends.size() << endl;
}

void Person::addFriend(const Person& friendPerson) {
    friends.push_back(friendPerson);
}

