import os
import streamlit as st
import subprocess

def main():
    st.title("YouTube Playlist Downloader")

    # Download Directory

    # fetch songs in playlist url
    playlist_url = st.text_input("Enter YouTube Playlist URL:")
    if playlist_url not in st.session_state and playlist_url:
        st.session_state.playlist_url = playlist_url

    st.sidebar.header("Playlist Songs")
    songs = subprocess.check_output(f"youtube-dl --flat-playlist --skip-download --get-title {st.session_state.playlist_url}", shell=True, text=True)
    song_list = songs.split('\n')[:-1]
    selected_items = st.sidebar.checkbox("Select All", key="select_all", value=True)
    for i, song in enumerate(song_list):
        checkbox_key = f"checkbox_{song}"
        st.sidebar.checkbox(f"{i+1}. {song}", key=checkbox_key, value=selected_items)

        # Display the list of songs in the sidebar
        #  selected_songs = st.multiselect("Select Songs to Download", song_list)

    if selected_items:
        st.write(selected_items)


if __name__ == "__main__":
    main()

