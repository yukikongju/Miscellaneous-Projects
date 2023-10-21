#include <PersonBernouilli.h>
#include <Person.h>

PersonBernouilli::PersonBernouilli(int id, double probability): Person(id), p(probability) {}

void PersonBernouilli::printInfos() {
    cout << "id: " << id << "; proba: " << p << "; hasInformation: " << hasInformation << "; number of friends: " << friends.size() << endl;
}

