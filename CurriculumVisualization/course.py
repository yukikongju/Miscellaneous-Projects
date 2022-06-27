#!/env/bin/python

import requests
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os

from bs4 import BeautifulSoup

from constants import BachelorURL


class UndergraduateProgram(object):

    def __init__(self, base_url, name):
        self.base_url = base_url
        self.download_path_base = "UndergraduatePrograms/"
        self.name = name
        self.filename = f'{self.download_path_base}{name}.csv'
        print(self.filename)

        # fetch courses if csv doesnt exist
        os.makedirs(self.download_path_base, exist_ok=True)
        if not os.path.isfile(self.filename):
            self.df = self.fetch_all_courses()
        else: 
            self.df = pd.read_csv(self.filename)

    def fetch_all_courses(self):
        """ Fetch all courses in program
            return: [sigle, title, credit, url, prerequisites]
        """

        courses = []

        url = f"{self.base_url}structure-du-programme/"

        # find all courses in programs
        r = requests.get(url)
        if r.status_code == 200: # requests is succesful
            soup = BeautifulSoup(r.content, 'html5lib')
            #  courses_body = soup.find_all('th', {'class': 'col-course'})
            courses_body = soup.find_all('tbody', {'class': 'programmeCourse fold'})
            print(len(courses_body))

            for tr in courses_body:
                sigleTag = tr.find('th', {'class': 'col-course'}).find('a')
                if sigleTag:
                    sigle = sigleTag.text.strip()
                    url_postfix = tr.find('th', {'class': 'col-course'}).find('a')['href']
                    course_url = f"https://admission.umontreal.ca{url_postfix}"
                    title = tr.find('td', {'class': 'col-title'}).text
                    credit = tr.find('td', {'class': 'col-credit'}).text

                    courses.append([sigle, title, credit, course_url])

        # convert array to dataframe
        df = pd.DataFrame(courses, columns=['Sigle', 'Titre', 'Credit', 'url']) 

        # find préalables if any
        prealables = []
        for url in df['url']:
            print(url)
            prealable = Course(url).get_prerequisites()
            if prealable:
                prealables.append(prealable[0])
            else: 
                prealables.append('')
            print(prealable)

        # save courses as csv
        df['Prerequisites'] = prealables
        df.to_csv(f'{self.download_path_base}{self.name}.csv', sep=',', index=False)
        df.drop_duplicates()

        return df

    def draw_curriculum_graph(self): # TODO
        G = nx.DiGraph(program = self.name)

        #  G.add_node('Data Structures', sigle='IFT2015', title='Data Structures')
        #  G.add_node('Algorithmes', sigle='IFT2125', title='Algorithmes')

        # add nodes to graph
        # 1. add course node
        for i, row in self.df.iterrows():
            titre = row['Titre']
            sigle = row['Sigle']
            G.add_node(sigle, Sigle=sigle)
        # TODO: 2. add arcs if course has prerequisites
        for i, row in self.df.iterrows():
            titre = row['Titre']
            sigle = row['Sigle']
            prerequisites = row['Prerequisites']
            for prereq in str(prerequisites).split(','):
                if prereq != 'nan': # if prereq is not nan
                    G.add_edge(prereq, sigle)

        # draw graph with labels
        labels = nx.get_node_attributes(G, 'Sigle')
        nx.draw(G, labels=labels)
        plt.show()

        
class Course(object):

    def __init__(self, url):
        self.url = url
        #  self.title = title
        #  self.sigle
        #  self.credit
        #  self.prerequisites
        pass

    def get_prerequisites(self):
        r = requests.get(self.url)
        soup = BeautifulSoup(r.content, 'html5lib')
        exigences = soup.find('div', attrs={'class': 'sommaireCol attributes'})
        prerequisites = [prereq.text.replace('Préalable: ', '').replace('Préalables: ', '').strip() for prereq in exigences.find_all('p', attrs={'class': 'specDefinition'}) if 'Préalable' in prereq.text]
        return prerequisites

class Informatique(UndergraduateProgram):

    def __init__(self):
        UndergraduateProgram.__init__(self, 'https://admission.umontreal.ca/programmes/baccalaureat-en-informatique/', 'Informatique')

class Mathematiques(UndergraduateProgram):

    def __init__(self):
        UndergraduateProgram.__init__(self, 'https://admission.umontreal.ca/programmes/baccalaureat-en-mathematiques/', 'Mathématiques')

class Bioinformatique(UndergraduateProgram):

    def __init__(self):
        UndergraduateProgram.__init__(self, 'https://admission.umontreal.ca/programmes/baccalaureat-en-bio-informatique/', 'Bioinformatique')

class Demographie(UndergraduateProgram):

    def __init__(self):
        UndergraduateProgram.__init__(self, 'https://admission.umontreal.ca/programmes/baccalaureat-en-demographie-et-statistique/', 'Démographie')

class Politique(UndergraduateProgram):

    def __init__(self):
        UndergraduateProgram.__init__(self, 'https://admission.umontreal.ca/programmes/baccalaureat-en-science-politique/', 'Politique')

class Histoire(UndergraduateProgram):

    def __init__(self):
        UndergraduateProgram.__init__(self, 'https://admission.umontreal.ca/programmes/baccalaureat-en-histoire/', 'Histoire')

class Philosophie(UndergraduateProgram):

    def __init__(self):
        UndergraduateProgram.__init__(self, 'https://admission.umontreal.ca/programmes/baccalaureat-en-philosophie/', 'Philosophie')

class Chimie(UndergraduateProgram):

    def __init__(self):
        UndergraduateProgram.__init__(self, 'https://admission.umontreal.ca/programmes/baccalaureat-en-chimie/', 'Chimie')

class GeographieEnvironnementale(UndergraduateProgram):

    def __init__(self):
        UndergraduateProgram.__init__(self, 'https://admission.umontreal.ca/programmes/baccalaureat-en-geographie-environnementale/', 'Géographie Environnementale')

        
def main():
    programme = Informatique()
    programme.draw_curriculum_graph()

if __name__ == "__main__":
    main()
