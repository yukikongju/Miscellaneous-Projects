import youtube_dl
import pandas as pd
import subprocess
import streamlit as st

def get_playlist_info(playlist_url):
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
    df = df[['title', 'uploader', 'url']]

    return df

def download_selected_songs(selected_songs: [str]):
    #  selected_songs = df.iloc[start_index-1:end_index]

    for _, song_url in selected_songs.enumerate():
        download_command = [
            'youtube-dl',
            '-x',
            '--audio-format', 'mp3',
            '--audio-quality', '0',
            '--embed-thumbnail',
            '--add-metadata',
            '-o', f"%(uploader)s - %(title)s.%(ext)s",
            '--match-title', f"^(?!(.* - Topic)).*",
            '--download-archive', 'archive.txt',
            song_url
        ]
        subprocess.run(download_command)

def main():
    # Example playlist URL
    youtube_playlist = 'https://www.youtube.com/playlist?list=PLzx7xtGqjNzoElKjq_zmpzgoS8RNrqHYh'

    # Fetch playlist information and create a DataFrame
    playlist_df = get_playlist_info(youtube_playlist)

    # Display the playlist information
    print("Playlist Information:")
    print(playlist_df)

    # Example: Download selected songs from index 1 to 3
    start_index = 1
    end_index = 3
    download_selected_songs(playlist_df, start_index, end_index)

if __name__ == "__main__":
    main()

