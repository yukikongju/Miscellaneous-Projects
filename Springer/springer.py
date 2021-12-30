#!/env/bin/python

import pandas as pd
import numpy as np
import requests
import os

from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC 
from selenium.common.exceptions import TimeoutException

class SpringerBook(object):

    def __init__(self, url):
        self.url = url

    def download_book(self):
        pass

    def get_book_information(self):
        """ get info: title, author, price, is_downloadable, author info, chapter titles"""

        r = requests.get(self.url)
        if r.status_code == 200:
            soup = BeautifulSoup(r.content, 'html5lib')

            # get title and subtitle
            title = soup.find('h1', attrs={'itemprop':'name'}).text
            subtitle = soup.find('h2', attrs={'class':'page-title__subtitle'}).text

            # get authors and about author
            authors = soup.find('span', attrs={'class': 'authors__name'}).text
            about_authors = soup.find('div', attrs={'id': 'about-the-authors'}).find('div').text

            # get book type
            book_type = soup.find('span', attrs={'id': 'content-type'}).text

            # get book series
            book_series = soup.find('div', attrs={'class': 'vol-info u-interface'}).find('a').text

            # get number of downloads
            num_downloads  = soup.find('span', attrs={'class': 'test-metric-count article-metrics__views'}).text


            # TODO: get prices for ebook, hard cover, soft cover
            # TODO: get ISBN for ebook, hard cover, soft cover
            # TODO: check if book is downloadable

            # get chapter titles
            chapters_item = soup.find_all('li', attrs={'class': 'chapter-item content-type-list__item'})
            chapters_info = []
            for item in chapters_item:
                chapter_title = item.find('a', attrs={'class': 'content-type-list__link u-interface-link'}).text
                #  chapter_author = item.find('div', attrs={'data-test': 'authors-text'}).text
                page_range = item.find('span', attrs={'data-test': 'page-range'}).text.replace('Pages ', '')
                chapters_info.append([chapter_title, page_range])

        
        return [title, subtitle, authors, book_type, book_series, num_downloads, 
                chapters_info]

    def download_book_info_as_excel(self): # TODO
        pass



class SpringerBookSeries(object):

    """ Download Complete Book Series """


    def __init__(self, ID):
        self.ID = ID
        self.url = f"https://www.springer.com/series/{ID}" 

    def save_series_as_csv(self):
        """ Fetch List of books in the series (dataframe: title, author, link) """

        # get series name
        r = requests.get(self.url)
        soup = BeautifulSoup(r.content, 'html5lib')
        serie_title = soup.find("h1", attrs={"class": "c-product-header__title u-word-wrap"}).find('a').get_text()
        print(serie_title)

        # get series_description
        #  serie_description = soup.find('div', attrs={'class': 'app-promo-text app-promo-text--keyline'}).find('p').get_text()
        #  print(serie_description)

        # iterate through all pages if request is successful
        end = False
        index = 1
        books = []
        while not end:
            page_url = self.url + f"/books?page={index}"
            r = requests.get(page_url)
            print(f"Fetching Page {index}")
            if (r.status_code) != 200:
                end = True
                break
            else: 
                soup = BeautifulSoup(r.content, 'html5lib')
                # find books in series: book[title, author, link]
                books_divs = soup.find_all("article", {"class": "c-card c-card--flush u-flex-direction-row"})
                for article in books_divs:
                    title_div = article.find("h3", {"class":"c-card__title"})
                    title = title_div.find("a").text.strip()
                    link = title_div.find("a")['href']
                    authors_ul = article.find("ul", {"class": "u-display-inline c-author-list c-author-list--compact c-author-list--book-series u-text-sm"})
                    authors = [span.find("span").text for span in authors_ul.find_all("li")]

                    # TODO: make requests on book page


                    book = [title, authors, link]
                    books.append(book)

                index += 1

        # create dataframe for series
        df = pd.DataFrame(books, columns=['Title', 'Authors', 'Link'])

        # save dataframe as csv if doesnt exist
        os.makedirs(f'{serie_title}', exist_ok=True)
        if not os.path.exists(f'{serie_title}/{serie_title}.xlsx'):
            #  df.to_excel(f'{serie_title}/{serie_title}.xlsx',index=True)
            df.to_excel(f'{self.base_path}/{serie_title}.xlsx',index=True)


    def download_book_series(self):
        """ Download complete book series """
        pass


def main():
    #  springer = SpringerBookSeries(3423) # Undergraduate Mathematics Series
    #  springer = SpringerBookSeries(666) # Undergraduate Texts in Mathematics 
    #  springer = SpringerBookSeries(3464) # Undergraduate Texts in Computer Science
    #  springer = SpringerBookSeries(7592) # Undergraduate Topics in Computer Science
    #  springer = SpringerBookSeries(15681) # Springer Actuarial
    #  springer = SpringerBookSeries(714) # Problem Books in Mathematics
    #  springer = SpringerBookSeries(10099) # Springer Texts in Business and Economics
    #  springer = SpringerBookSeries(11225) # Compact Textbooks in Mathematics
    #  springer = SpringerBookSeries(13205) # Probability Theory and Stochastic Modeling
    #  springer = SpringerBookSeries(4129) # Advances in Mathematical Economics
    #  springer = SpringerBookSeries(15561) # Foundations for Undergraduate Research in Mathematics
    #  springer = SpringerBookSeries(4318) # CMS Books in Mathematics
    #  springer = SpringerBookSeries(4893) # Probability and its Applications
    #  springer = SpringerBookSeries(7879) # EAA Series
    #  springer = SpringerBookSeries(1560) # Probability and its Applications
    #  springer = SpringerBookSeries(602) # Stochastic Modelling and Applied Probability
    #  springer = SpringerBookSeries(13083) # Mathematical Biosciences Institute Lecture Series
    #  springer = SpringerBookSeries(632) # Lecture Notes in Chemistry
    #  springer = SpringerBookSeries(8819) # Theory and Applications of Computability
    #  springer = SpringerBookSeries(4752) # Information Security and Cryptography
    #  springer = SpringerBookSeries(4190) # Natural Computing Series
    #  springer = SpringerBookSeries(4097) # Springer Praxis Books
    #  springer = SpringerBookSeries(8158) # Popular Science
    #  springer = SpringerBookSeries(3464) # 
    #  springer = SpringerBookSeries(3464) # 
    #  springer = SpringerBookSeries(3464) # 
    #  springer.save_series_as_csv()

    #  book = SpringerBook("https://link.springer.com/book/10.1007/978-3-030-00632-7")
    #  book.download_book()

    book = SpringerBook('https://link.springer.com/book/10.1007/978-3-030-44074-9')
    info = book.get_book_information()



if __name__ == "__main__":
    main()

