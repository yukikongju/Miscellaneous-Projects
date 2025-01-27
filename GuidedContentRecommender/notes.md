# Todos

**Getting the data and preprocessing**

- [X] 0. Download all google docs from Google Drive and put it into `raw_data` directory
	- [X] [SleepTales](https://drive.google.com/drive/folders/1qdk0Su_vD7hXMlK44wG3p2-EPpb53K8j)
	- [X] [Meditations](https://drive.google.com/drive/folders/1caTUsbK7GBT5-ec4hNX7GL9dnxJNh7se?q=type:document%20parent:1caTUsbK7GBT5-ec4hNX7GL9dnxJNh7se)
- [O] 1. Preprocess word documents for meditations
	- [X] delete all word documents that is not english
	- [X] rename word documents without spaces with `rename -n s/ /_/g`
	- [X] convert word documents (.docx) to text files (.txt) with `pandoc -s <file>.docx -o <file>.txt`; see [scripts/convert_word_to_txt.sh]
	- [ ] Cleanup individual files
- [ ] Preprocess word documents for sleeptales
- [ ] 2. Upload clean files to google drive to be downloaded later

**Creating the Embedding**



**Making the streamlit demo**
