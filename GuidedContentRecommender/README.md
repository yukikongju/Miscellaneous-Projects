# Guided Content Recommender

Content Recommender:
- [X] Content-based filtering => recommendation based on content metadata, audio/text, ..
    * Ex: vectorize all content based on their text/audio using Doc2Vec and
      use KNN to find the closest recommendation
- [ ] Collaborative Filtering => use users similar content to suggest


Making an Embedding for Guided content based on text. Trying to implement
the following:

**Document to Vector Embedding**

- [X] Doc2Vec with [gensim](https://tedboy.github.io/nlps/generated/generated/gensim.models.Doc2Vec.html)
- [ ] Matrix Factorization with [pytorch](https://d2l.ai/chapter_recommender-systems/mf.html)
- [ ] Collaborative Filtering with [autoencoder architecture](https://d2l.ai/chapter_recommender-systems/autorec.html)
- [ ] [Pointwise approaches and Personalized Ranking](https://d2l.ai/chapter_recommender-systems/ranking.html)
- Universal Sentence Encoder (USE)
- Language Agnostic Sentence Representation (LASER)
- Sentence BERT
- T5 Embedding
- InferSent (?)

**Using word Embedding to create a document embedding**

- Method 1: Average Word Embedding
    * compute the average vector of all words embeddings, weighted by their
      importance using TF-IDF

**How to measure embedding performance?**


# Resources

- [Document Embedding Tutorial](https://github.com/Yeema/ecommerce-review-score-classification/blob/master/resources/docs/TUTORIAL_5_DOCUMENT_EMBEDDINGS.md)
- [Stack Overflow - Selecting right latent space dimension](https://www.reddit.com/r/MachineLearning/comments/urhj10/d_how_to_choose_dimensions_for_latent_space/)
- [Umap](https://www.reddit.com/r/MachineLearning/comments/urhj10/d_how_to_choose_dimensions_for_latent_space/)
