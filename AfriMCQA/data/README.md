# Datasets

- [X] [images](https://drive.google.com/drive/u/2/folders/1wNhnOZo-lkTz0o8ErUqg-dI0H0Cl3vv5)
- [X] [text-only](https://docs.google.com/spreadsheets/d/1dnhmdw6HRrncJsJOgemBJIsgvduuUuSCXT_9_qPMBig/edit?gid=1428962976#gid=1428962976)
- [X] [clean_dataset](https://docs.google.com/spreadsheets/d/1LEBi5GHktTjRn4noofnHAVEKtlRAadyZb1vLVa5tTRM/edit?pli=1&gid=1217658011#gid=1217658011)
- [X] [Custom Afri-MCQA](https://drive.google.com/drive/u/2/folders/178CX-o3OdY0vYh0M1xrTz-SNOG6cRWEj)


Notes:
- need to rename "native_question" from "ImageOnly" => `question_col = "eng_question" if lang == "english" else "native_question"`


> uv run python experiments/experiment_scoring.py --model-name gpt-4o --experiment-setup TextOnly --no-llm-judge --force

> csvsql --query "SELECT ID, Country, Language, Category, native_question AS eng_question, answer, answer_yoruba, answer_english, image_pat FROM stdin" your_file.csv > updated.csv
