import os
import streamlit as st
import subprocess
import youtube_dl
import pandas as pd
import time

def get_download_time_and_size(duration: int, media_type: str = 'mp3'):
    """
    Compute download time and size given mp3 duration in seconds

    Parameters
    ----------
    duration: int
        in seconds

    > Download time = file size (bytes) / download speed (bytes per seconds)
    > file size (byte) = bitrate x duration in seconds / 8
    > bitrate = download speed = frame size x frames rate
    """
    pass


@st.cache_data
def get_playlist_info(playlist_url: str):
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,
        'force_generic_extractor': True,
        'extractor_args': {'youtube:playlist': {'skip_download': True}},
        'format': 'bestvideo+bestaudio/best',  # Add the format option
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(playlist_url, download=False)
        entries = result['entries']

    # Create a DataFrame with artist name, song title, and URL
    df = pd.DataFrame(entries)
    df = df[['title', 'uploader', 'id', 'duration']]

    return df

def download_selected_songs(df):
	#  youtube-dl -x --audio-format mp3 --audio-quality 0 --embed-thumbnail --add-metadata -o "%(uploader)s - %(title)s.%(ext)s" --match-title "^(?!(.* - Topic)).*" --download-archive archive.txt $(youtube_playlist)

    # TODO: remove " - Topic from title"


    progress_bar = st.progress(0, text="Download in Progress")

    for i, song in df.iterrows():
        song_url = f"https://www.youtube.com/watch?v={song['id']}"
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

        # TODO: progress bar
        progress_bar.progress(i/len(df), text = f"Downloading {song['title']} {i}/{len(df)}")
        subprocess.run(download_command)
    time.sleep(1)
    progress_bar.empty()

def dataframe_with_selections(df):
    df_with_selections = df.copy()
    df_with_selections.insert(0, "Select", st.session_state.checkbox_init_value)

    # Get dataframe row-selections from user with st.data_editor
    edited_df = st.data_editor(
        df_with_selections,
        hide_index=True,
        column_config={"Select": st.column_config.CheckboxColumn(required=True)},
        disabled=df.columns,
    )

    # Filter the dataframe using the temporary column, then drop the column
    selected_rows = edited_df[edited_df.Select]
    return selected_rows.drop('Select', axis=1)



def main():
    # 1. create streamlit widgets: (1) playlist url (2) Download Directory
    st.title("YouTube Playlist Downloader")

    # TODO: folder selector directory

    # 2. fetch urls of playlist once
    #  youtube_playlist = 'https://www.youtube.com/playlist?list=PLzx7xtGqjNzoElKjq_zmpzgoS8RNrqHYh'
    playlist_url = st.text_input("Enter YouTube Playlist URL:")
    if playlist_url:
        df_playlist = get_playlist_info(playlist_url=playlist_url)
        df_selection = df_playlist.copy()

        container = st.container(border=True)
        # button: Select All FIXME: weird behavior
        st.session_state.checkbox_init_value = False
        if container.button(label="Select All"):
            st.session_state.checkbox_init_value = True
        if container.button(label="Unselect All"):
            st.session_state.checkbox_init_value = False

        df_selection = dataframe_with_selections(df_playlist)

        # FIXME: compute download time from songs duration
        download_time = df_selection['duration'].sum() / 6
        download_time_min, download_time_sec = download_time // 60 , download_time % 60
        st.write(f"Estimated Download Time: **{download_time_min} min {download_time_sec} s**")

        # FIXME: compute selection size from songs duration
        FPS = 25
        FRAME_LENGTH = 640 # 1152
        download_size = df_selection['duration'].sum() * FPS * FRAME_LENGTH
        download_size_mib = round(download_size / 1_000_000, 4)
        st.write(f"Estimated Download Size: **{download_size_mib} MiB**")

        # 4. download songs if button is pressed
        if st.button(label="Download"):
            download_selected_songs(df_selection)


if __name__ == "__main__":
    main()

