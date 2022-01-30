#!/usr/bin/env python

import requests
import wikipedia
import os

from googlesearch import search
from youtubesearchpython import PlaylistsSearch


class Searcher(object):

    def __init__(self, download_path:str, subject:str):
        self.subject = subject
        self.download_path = download_path

        # check if download_path is_valid, if it is not, create it
        os.makedirs(f'{self.download_path}', exist_ok=True)


    def _get_top_k_google_search(self, query:str, n=20):
        """ return list of top n links found from googling query 
            query: what to search
            n: num playlists
        """
        return [url for url in search(query, stop=n, pause=0.3)]    # add pause to avoid IP address being blocked
        
    def _get_youtube_playlist(self, query:str, n:int):
        """ course = [channel, title, url]
            query: what to search
            n: num playlists
        """
        courses = []
        search = PlaylistsSearch(query, limit=n).result()
        for playlist in search['result']:
            channel = (playlist['channel'])['name']
            title = playlist['title']
            url = playlist['link']
            #  courses.append([channel, title, url])
            courses.append({"Channel": channel, "Title": title, "Link": url})
        return courses 

    def _get_wiki_search(self, query):
        """ get wikipedia search on query
            query: what to search
        """
        wikipedia.set_lang('en')
        summary = wikipedia.summary(self.subject, sentences = 3)
        subject_url = wikipedia.page(self.subject).url
        topics_url = wikipedia.page(f'List of {self.subject} topics').url

        return summary, subject_url, topics_url


    def _get_ressources(self):
        pass
        
    def _get_textbooks(self, n:int):
        """ get best textbook by checking links from quora and reddit suggestion """

        # 1. get quora and reddit links for textbooks suggestions
        search_results = self._get_top_k_google_search(f'{self.subject} best textbooks', n)

        # 2. TODO: go inside each links and add all urls to textbooks
        textbooks_urls = []
        for url in search_results: 
            textbooks_urls.append(url)

        # 3. TODO: remove dupplicates

        return textbooks_urls

    def _get_urls_in_page(self, url): # TODO
        """ Get all url inside page """
        r = requests.get(url)
        # get all links inside page
        if r.status_code == 200: # success
            pass


    def create_document(self):
        """ create markdown file for youtube playlist, course, textbook, 
        problem set, solution, ... """

        # 1. get ressources: youtube course, pdfs

        print('Searching for Topics on wikipedia...')
        wiki_summary, wiki_url, wiki_topics_url = self._get_wiki_search(f'{self.subject} Topics')
        print('Found.')

        print('Searching for Youtube Courses...')
        youtube_playlists = self._get_youtube_playlist(f'{self.subject} course', 30)
        print('Found Youtube Courses.')

        print('Searching for Online Courses...')
        online_courses = self._get_top_k_google_search(f'{self.subject} courses', 25)
        print('Found Youtube Courses.')


        print('Searching for pdfs, lecture notes and problem set ...')
        pdfs = self._get_top_k_google_search(f'{self.subject} pdf', 25)
        lecture_notes = self._get_top_k_google_search(f'{self.subject} lecture notes', 25)
        slides = self._get_top_k_google_search(f'{self.subject} slides', 25)
        solutions = self._get_top_k_google_search(f'{self.subject} solutions', 25)
        print('Found.')

        print('Searching for Textbooks')
        textbooks = self._get_textbooks(25)
        print('Found.')

        # 2. write file

        print('Generating File...')

        with open(f'{self.download_path}/{self.subject}.md', 'w') as f:
            # write title
            f.write(f'# {self.subject} Ressources\n\n')

            # Summary
            f.write('**Summary**\n\n')
            f.write(f'{wiki_summary}\n\n')

            # Wikipedia
            f.write('**Wikipedia**\n\n')
            f.write(f'- [{self.subject} Wiki]({wiki_url})\n')
            f.write(f'- [{self.subject} Wiki Topics]({wiki_topics_url})\n')


            # Youtube
            f.write('\n**Youtube Courses**\n\n')
            for youtube_playlist in youtube_playlists:
                title = youtube_playlist['Title']
                link = youtube_playlist['Link']
                channel = youtube_playlist['Channel']
                f.write(f'- [{title} - {channel}]({link})\n')

            # Google Search
            f.write('\n**Online Courses**\n\n')
            for course in online_courses:
                f.write(f'- {course}\n')

            f.write('\n**PDFs**\n\n')
            for pdf in pdfs:
                f.write(f'- {pdf}\n')
            
            f.write('\n**Lecture Notes**\n\n')
            for note in lecture_notes:
                f.write(f'- {note}\n')

            f.write('\n**Slides**\n\n')
            for slide in slides:
                f.write(f'- {slide}\n')

            f.write('\n**Solutions**\n\n')
            for sol in solutions:
                f.write(f'- {sol}\n')
        
            f.write('\n**Textbooks Suggestions**\n\n')
            for textbook in textbooks:
                f.write(f'- {textbook}\n')

            print('Completed!')


def main():
    searcher = Searcher('Ressources-Searcher','Embedded Systems')
    searcher.create_document()
    #  textbooks = searcher._get_textbooks(20)
    

if __name__ == "__main__":
    main()


