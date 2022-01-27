import requests

from googlesearch import search
from youtubesearchpython import PlaylistsSearch
#  from google import google

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
            courses.append([channel, title, url])
        return courses 

        
    def create_document(self):
        """ create markdown file for youtube playlist, course, textbook, 
        problem set, solution, ... """

        # 1. get ressources: youtube course, pdfs

        print('Searching for Youtube Courses...')
        youtube_playlists = self._get_youtube_playlist(f'{self.subject} course', 25)
        print('Found Youtube Courses.')

        print('Searching for pdfs and lecture notes ...')
        #  pdfs = self._get_top_k_google_search(f'{self.subject} pdf', 20)
        lecture_notes = self._get_top_k_google_search(f'{self.subject} lecture notes', 20)
        slides = self._get_top_k_google_search(f'{self.subject} slides', 20)


        # 2. write file

        


def main():
    searcher = Searcher('Complex Analysis')
    searcher.create_document()
    

if __name__ == "__main__":
    main()


