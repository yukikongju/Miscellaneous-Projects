import os
import streamlit as st
import subprocess
import youtube_dl
import pandas as pd



@st.cache_data
def get_playlist_info(playlist_url: str):
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,
        'force_generic_extractor': True,
        'extractor_args': {'youtube:playlist': {'skip_download': True}},
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(playlist_url, download=False)
        entries = result['entries']

    # Create a DataFrame with artist name, song title, and URL
    df = pd.DataFrame(entries)
    df = df[['title', 'uploader', 'id']]

    return df

def download_selected_songs(songs_ids: [str]):
	#  youtube-dl -x --audio-format mp3 --audio-quality 0 --embed-thumbnail --add-metadata -o "%(uploader)s - %(title)s.%(ext)s" --match-title "^(?!(.* - Topic)).*" --download-archive archive.txt $(youtube_playlist)

    # TODO: remove " - Topic from title"

    for i, song_id in enumerate(songs_ids):
        song_url = f"https://www.youtube.com/watch?v={song_id}"
        download_command = [
            'youtube-dl',
            '-x',
            '--audio-format', 'mp3',
            '--audio-quality', '0',
            '--embed-thumbnail',
            '--add-metadata',
            '-o', f"%(uploader)s - %(title)s.%(ext)s",
            #  '--match-title', f"^(?!(.* - Topic)).*",
            '--download-archive', 'archive.txt',
            song_url
        ]

        #  progress_bar = st.progress(0, text="Downloading ")
        subprocess.run(download_command)


def main():
    # 1. create streamlit widgets: (1) playlist url (2) Download Directory
    st.title("YouTube Playlist Downloader")

    # 2. fetch urls of playlist once
    #  youtube_playlist = 'https://www.youtube.com/playlist?list=PLzx7xtGqjNzoElKjq_zmpzgoS8RNrqHYh'
    playlist_url = st.text_input("Enter YouTube Playlist URL:")
    if playlist_url:
        df_playlist = get_playlist_info(playlist_url=playlist_url)

        container = st.container(border=True)
        # button: Select All
        checkbox_init_value = True
        if container.button(label="Select All"):
            checkbox_init_value = True
        if container.button(label="Unselect All"):
            checkbox_init_value = False

        #
        cols = container.columns([2, 1])

        # 3. select songs urls with checkboxes in a container


        checkbox_keys = {}
        for i, song in df_playlist.iterrows():
            checkbox_key = f"checkbox_{song['id']}"
            checkbox_keys[song['id']] = cols[0].checkbox(label=f"{i+1} - {song['title']}", key=checkbox_key, value=checkbox_init_value)


        # get selected items
        selected_songs_ids = [key for key, value in checkbox_keys.items() if value]
        cols[1].write(selected_songs_ids)


        # 4. download songs if button is pressed
        if st.button(label="Download"):
            download_selected_songs(songs_ids=selected_songs_ids)

            # TODO: progress bar


    pass


if __name__ == "__main__":
    main()

