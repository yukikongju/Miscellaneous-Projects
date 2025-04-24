# Name Entity Recognition (NER)

Uses Cases:
- Knowledge Graph Builder
- NER for Audio Transcript
- Historical Entity Recognition
- Email Assistand => detect names, dates, times, places, actions in emails to create calendar events
- News Stream Classifier => Wrapper around RSS feed to get articles related to X
- Legal Document Analyzer => Extract legal entities like case numbers, judges, plaintiffs, defendants, laws referenced, etc.


## Steps

1. Problem Definition => what entities are we trying to recognize
2. Data Collection => the dataset + its tags
    a. Augment the dataset
3. Preprocessing
    a. Tokenization => tokenize sentence + Embedding (optional)
    b. Encoding Labels =>
4. Build and Train the Model

## Models

- Conditional Random Fields (CRF)
- Spacy
- Bi-directional LSTM + CRF
- Transformers (BERT, RoBERTa)

```
python -m spacy download en_core_web_sm
```

## Resources

- [Kaggle Dataset](https://www.kaggle.com/datasets/naseralqaydeh/named-entity-recognition-ner-corpus?utm_source=chatgpt.com)
