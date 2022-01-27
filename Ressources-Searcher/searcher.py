import requests

from googlesearch import search
from youtubesearchpython import PlaylistsSearch
#  from google import google

class Searcher(object):

    def __init__(self, subject:str):
        self.subject = subject

        # create and save document
        self.create_document()
        

    def _get_top_k_google_search(self, k:int, query:str):
        """ return list of top k links found from googling query """
        #  return [url for url in search(query, tld="co.in", num=10, stop=k, lang='en')]
        pass
        
    def _get_pdf_urls(self, k:int, query:str):
        # TODO: get pdfs url
        #  pdfs = self._get_pdf_urls(10, f'{self.subject} pdf')

        # TODO: trier selon best fit
        pass

    def _get_youtube_playlist(self):
        """course = [channel, title, url]"""
        courses = []
        search = PlaylistsSearch(f'{self.subject} course', limit=25).result()
        for playlist in search['result']:
            channel = (playlist['channel'])['name']
            title = playlist['title']
            url = playlist['link']
            courses.append([channel, title, url])
        return courses 

        
    def create_document(self):
        """ create markdown file for youtube playlist, course, textbook, 
        problem set, solution, ... """

        # get ressources: youtube course, pdfs
        youtube_playlists = self._get_youtube_playlist()
        #  pdfs = self._get_pdf_urls(10, f'{self.subject} pdf')

        


def main():
    searcher = Searcher('Complex Analysis')
    searcher.create_document()
    

if __name__ == "__main__":
    main()


