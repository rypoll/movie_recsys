# A Movie Recommender

## Movie recommendations based on a movie you like.

This is an app created using Streamlit, deployed using docker and Google Cloud Run that uses an IMDB movies dataset from [Kaggle](https://www.kaggle.com/rounakbanik/the-movies-dataset) and allows the user to input a movie to find similar movies that the user might enjoy. Not all movies are present in this app due to lack of local computing power but feel free to try "interstellar" as a test!

## How are movies recommended based off an inputted movie?
Each movie in the data contains a synopsis - a description of the plot of the movie. Movies also contain meta-data which captures things about the movie such as the director, the cast, and the genre of the movie. Using Natural Language Processing and Cosine Similarity, the synopsis and the meta-data for the inputted movie is compared the to synopses and meta-data of the other movies in the dataset. A score is given for each movie that indicates the degree of similarity between the movies, with higher scores being more similar. Factored into the similarity score is also the average rating and the number of votes of each of the movies - such that more popular and highly rated movies are more likely to appear in the recommendations.

## The app
The app is [here.](https://moviereco-gdkfo3moia-ew.a.run.app/)


