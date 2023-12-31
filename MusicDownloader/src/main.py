import os
import streamlit as st
import subprocess
import youtube_dl
import pandas as pd


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
        subprocess.run(download_command)


def main():
    # 1. create streamlit widgets
    youtube_playlist = 'https://www.youtube.com/playlist?list=PLzx7xtGqjNzoElKjq_zmpzgoS8RNrqHYh'

    # 2. fetch urls of playlist once
    df_playlist = get_playlist_info(youtube_playlist)


    # 3. select songs urls with checkboxes



    # 4. download songs if button is pressed
    download_selected_songs(songs_ids=list(df_playlist['id']))


    pass


if __name__ == "__main__":
    main()

