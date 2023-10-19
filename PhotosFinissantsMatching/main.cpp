#include <iostream>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>
#include <set>

using namespace std;


struct Student {
    string prenom, nom, diplome;
    string matricule, programme, description, specialisation, dateDiplome,
	   dateConseilExecutif, statut;



    Student(string prenom, string nom, string diplome): prenom(prenom), nom(nom), diplome(diplome) {}
    Student(string prenom, string nom, string diplome, string matricule, 
	    string programme, string description, string specialisation, 
	    string dateDiplome, string dateConseilExecutif, string statut):
	prenom(prenom), nom(nom), diplome(diplome), matricule(matricule), 
	programme(programme), description(description), dateDiplome(dateDiplome), 
	specialisation(specialisation), dateConseilExecutif(dateConseilExecutif),
	statut(statut) {}
    

};


int main() {
    // read csv files: ppatrick.csv, diro.csv, last_year.csv
    ifstream filePatrick("files/ppatrick.csv"), fileDIRO("files/diro.csv");

    vector<Student> peoplePatrick, peopleDIRO;
    string line;
    while (getline(filePatrick, line)) {
	istringstream iss(line);
	string prenom, nom, diplome;

	if(getline(iss, prenom, ',') && getline(iss, nom, ',') && getline(iss, diplome, ',')) {
	    Student student(prenom, nom, diplome);
	    peoplePatrick.push_back(student);
	}
    }

    while (getline(fileDIRO, line)) {
	istringstream iss(line);
    string prenom, nom, diplome, matricule, programme, description, specialisation, dateDiplome, dateConseilExecutif, statut;

	if(getline(iss, matricule, ',') && getline(iss, nom, ',') && 
		getline(iss, prenom, ',') && getline(iss, programme, ',') && 
		getline(iss, description, ',') && getline(iss, specialisation, ',')
		&& getline(iss, dateDiplome, ',') && getline(iss, dateConseilExecutif, ',')
		&& getline(iss, statut, ',')) {
	    Student student(prenom, nom, "BAC", matricule, programme, 
		    description, specialisation, dateDiplome, dateConseilExecutif, 
		    statut);
	    peopleDIRO.push_back(student);
	}


    }


    // print
    // for (const Student &person: peopleDIRO) {
    //     cout << person.prenom;
    // }

    // 2. Find missing students
    set<pair<string, string>> setPatrick, setDIRO;
    set<pair<string, string>> setdifference1, setdifference2;
    for (const Student &p : peopleDIRO) setDIRO.insert(make_pair(p.prenom, p.nom));
    for (const Student &p: peoplePatrick) {
	if (p.diplome == "BAC") setPatrick.insert(make_pair(p.prenom, p.nom));
    }

    set_difference(setPatrick.begin(), setPatrick.end(), setDIRO.begin(), setDIRO.end(), inserter(setdifference1, setdifference1.begin()));

    cout << endl << "Elements in Patrick but not in DIRO [ in BAC ]" << endl; // check if these people are graduating this autumn 2023
    for (const pair<string, string> &p: setdifference1) cout << p.first << " " << p.second << endl;

    // ---

    cout << endl << "Elements in DIRO but not in Patrick " << endl; // check if these people have taken their graduation pics last year

    set_difference(setDIRO.begin(), setDIRO.end(), setPatrick.begin(), setPatrick.end(), inserter(setdifference2, setdifference2.begin()));
    for (const pair<string, string> &p: setdifference2) cout << p.first << " " << p.second << endl;

    return 0;
}


