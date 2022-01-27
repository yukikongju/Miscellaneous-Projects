#!/usr/bin/env python

import requests
import wikipedia

from googlesearch import search
from youtubesearchpython import PlaylistsSearch


class Searcher(object):

    def __init__(self, subject:str):
        self.subject = subject


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
        

    def create_document(self):
        """ create markdown file for youtube playlist, course, textbook, 
        problem set, solution, ... """

        # 1. get ressources: youtube course, pdfs

        print('Searching for Topics on wikipedia...')
        wiki_summary, wiki_url, wiki_topics_url = self._get_wiki_search(f'{self.subject} Topics')
        print('Found.')

        print('Searching for Youtube Courses...')
        youtube_playlists = self._get_youtube_playlist(f'{self.subject} course', 25)
        print('Found Youtube Courses.')

        print('Searching for pdfs and lecture notes ...')
        pdfs = self._get_top_k_google_search(f'{self.subject} pdf', 20)
        lecture_notes = self._get_top_k_google_search(f'{self.subject} lecture notes', 20)
        slides = self._get_top_k_google_search(f'{self.subject} slides', 20)
        solutions = self._get_top_k_google_search(f'{self.subject} solutions', 20)
        print('Found.')


        # 2. write file
        with open(f'{self.subject}.md', 'w') as f:
            # write title
            f.write(f'# {self.subject} Ressources\n\n')

            # Summary
            f.write('**Summary**\n\n')
            f.write(f'{wiki_summary}\n\n')

            # Wikipedia
            f.write('**Wikipedia**\n\n')
            f.write(f'- [{self.subject} Wiki]({wiki_url})\n')
            f.write(f'- [{self.subject} Wiki Topics]({wiki_topics_url})\n')


            # TODO: Youtube
            f.write('\n**Youtube Courses**\n\n')
            for youtube_playlist in youtube_playlists:
                title = youtube_playlist['Title']
                link = youtube_playlist['Link']
                channel = youtube_playlist['Channel']
                f.write(f'- [{title} - {channel}]({link})\n')

            # TODO: Google Search
            f.write('\n**PDFs**\n\n')
            for pdf in pdfs:
                f.write(f'- {pdf}\n')
            
            f.write('\n**Lecture Notes**\n\n')
            for note in lecture_notes:
                f.write(f'- {note}\n')

            f.write('\n**Slides**\n\n')
            for slide in slides:
                f.write(f'- {note}\n')

            f.write('\n**Solutions**\n\n')
            for sol in solutions:
                f.write(f'- {sol}\n')
        


def main():
    searcher = Searcher('Complex Analysis')
    searcher.create_document()
    

if __name__ == "__main__":
    main()


