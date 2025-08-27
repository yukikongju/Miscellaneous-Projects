# Guided Recommender System w/ Content Filtering

- Assumption: users that listen to content will like content similar themes
- Idea: Vectorize new content with Doc2Vec and find the top n suggestions
  based on nearest vector distance with KNN
- Use Cases:
    * "Because you listened to" Carousel
    * "You may also like" Carousel
- Pipeline
    * Everytime a new guided content is dropped into a Google Drive, the following
      jobs get triggered:
      1. Guided Content Vectorization: Vectorize every new content based on
	 their text and store in them in a vector database
      2. Similarity + Re-ranking:
	  a. For the new content, compute its similarity to all other content
	  b. For existing content, add similarity to the new content and sort them

- Model Evaluation
    * We can use the guided content metadata to evaluate cluster
- Constraints
    * about 1000 content only
    * New realeases occurs about once every 2 weeks

## Resources
