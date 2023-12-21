import requests
import streamlit as st
from bs4 import BeautifulSoup


def get_rss(youtube_channel: str):
    """
    Find RSS url from youtube channel

    Parameters
    ----------
    url: str
        > youtube channel link

    Ex: https://www.youtube.com/feeds/videos.xml?channel_id=UCbRP3c757lWg9M-U7TyEkXA

    """
    # make the request
    response = requests.get(youtube_channel)

    # find the RSS url
    BASE_RSS_URL = "https://www.youtube.com/feeds/videos.xml?channel_id="
    soup = BeautifulSoup(response.text, 'html.parser')
    href = next((link.get('href') for link in soup.find_all('link', href=True)
                 if link['href'].startswith(BASE_RSS_URL)), None)
    return href

def main():
    st.title("Youtube RSS Finder")

    # define youtube channel input
    youtube_channel = st.text_input("Enter Youtube Channel URL: ")

    # generate rss url if button clicked
    if st.button("Get RSS") and youtube_channel:
        rss_url = get_rss(youtube_channel)
        if rss_url:
            st.success(f"The RSS url is: {rss_url}")
        else:
            st.error("Failed to Retrieve RSS url. please check the link.")


if __name__ == "__main__":
    main()

