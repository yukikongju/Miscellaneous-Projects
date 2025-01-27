# Todos

**Getting the data and preprocessing**

- [X] 0. Download all google docs from Google Drive and put it into `raw_data` directory
	- [X] [SleepTales](https://drive.google.com/drive/folders/1qdk0Su_vD7hXMlK44wG3p2-EPpb53K8j)
	- [X] [Meditations](https://drive.google.com/drive/folders/1caTUsbK7GBT5-ec4hNX7GL9dnxJNh7se?q=type:document%20parent:1caTUsbK7GBT5-ec4hNX7GL9dnxJNh7se)
- [X] 1. Preprocess word documents for meditations + sleeptales
	- [X] delete all word documents that is not english
	- [X] rename word documents without spaces with `rename -n s/ /_/g`
	- [X] convert word documents (.docx) to text files (.txt) with `pandoc -s <file>.docx -o <file>.txt`; see [scripts/convert_word_to_txt.sh]
	- [X] Cleanup individual files
- [ ] 2. Upload clean files to google drive to be downloaded later

**Creating the Embedding**

- [X] doc2vec


**Making the streamlit demo**

- select closest content and highlight them when hovering on content



**Evaluating embedding performance**

Metrics:
- cluster metrics: Silouhette score, normalized mutual informantion (NMI)
