#Core Pkg
import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
#Load Our Dataset
def load_data(data):
    df = pd.read_csv(data)
    return df
#Import data from the clean file 

df = pd.read_csv('data/metadata_clean.csv')

#Print the head of the cleaned DataFrame
#df.head()


#Import the original file
orig_df = pd.read_csv('data/movies_metadata.csv', low_memory=False)

#Add the useful features into the cleaned dataframe
df['overview'], df['id'] = orig_df['overview'], orig_df['id']

df.head()

df = df.sample(frac=0.1, replace=False, random_state=1)
df.to_csv("data/df_sampled.csv")



#this is on a program because it stops it running apparently every time

def func(): 
    df = pd.read_csv('data/df_sampled.csv')
    df = df.reset_index(drop=True)
    #delete duplicates here 
    df = df.loc[df.astype(str).drop_duplicates().index]
    df = df.drop_duplicates()
    #st.write("This is effin running again!")
    # Function to convert all non-integer IDs to NaN
    def clean_ids(x):
        try:
            return int(x)
        except:
            return np.nan


    #Clean the ids of df
    df['id'] = df['id'].apply(clean_ids)

    #Filter all rows that have a null ID
    df = df[df['id'].notnull()]
    df = df.reset_index(drop=True)

    # Load the keywords and credits files
    cred_df = pd.read_csv('data/credits.csv')
    key_df = pd.read_csv('data/keywords.csv')

    # Function to convert all non-integer IDs to NaN
    def clean_ids(x):
        try:
            return int(x)
        except:
            return np.nan
            
            #Clean the ids of df
    df['id'] = df['id'].apply(clean_ids)

    #Filter all rows that have a null ID
    df = df[df['id'].notnull()]

    # Convert IDs into integer
    df['id'] = df['id'].astype('int')
    key_df['id'] = key_df['id'].astype('int')
    cred_df['id'] = cred_df['id'].astype('int')

    # Merge keywords and credits into your main metadata dataframe
    df = df.merge(cred_df, on='id')
    df = df.merge(key_df, on='id')

    #Display the head of df
    #df.head()

    df = df.sort_values(by=['vote_count'], ascending=False)

    df = df[~df.duplicated(subset=['title', 'year', 'id'])]

    df = df.reset_index(drop=True)

    #Import TfIdfVectorizer from the scikit-learn library
    from sklearn.feature_extraction.text import TfidfVectorizer

    #Define a TF-IDF Vectorizer Object. Remove all english stopwords
    tfidf = TfidfVectorizer(stop_words='english')

    #Replace NaN with an empty string
    df['overview'] = df['overview'].fillna('')

    #Construct the required TF-IDF matrix by applying the fit_transform method on the overview feature
    tfidf_matrix = tfidf.fit_transform(df['overview'])

    #Output the shape of tfidf_matrix
    #tfidf_matrix.shape

    # Import linear_kernel to compute the dot product
    from sklearn.metrics.pairwise import linear_kernel

    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    #Construct a reverse mapping of indices and movie titles, and drop duplicate titles, if any
    indices = pd.Series(df.index, index=df['title'].str.lower()).drop_duplicates()

    # Convert the stringified objects into the native python objects
    from ast import literal_eval

    features = ['cast', 'crew', 'keywords', 'genres']
    for feature in features:
        df[feature] = df[feature].apply(literal_eval)
        
        
    # Extract the director's name. If director is not listed, return NaN
    def get_director(x):
        for crew_member in x:
            if crew_member['job'] == 'Director':
                return crew_member['name']
        return np.nan

    #Define the new director feature
    df['director'] = df['crew'].apply(get_director)

    # Returns the list top 3 elements or entire list; whichever is more.
    def generate_list(x):
        if isinstance(x, list):
            names = [i['name'] for i in x]
            #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
            if len(names) > 3:
                names = names[:3]
            return names

        #Return empty list in case of missing/malformed data
        return []
        
    #Apply the generate_list function to cast and keywords
    df['cast'] = df['cast'].apply(generate_list)
    df['keywords'] = df['keywords'].apply(generate_list)

    #Only consider a maximum of 3 genres
    df['genres'] = df['genres'].apply(lambda x: x[:3])

    # Function to sanitize data to prevent ambiguity. It removes spaces and converts to lowercase
    def sanitize(x):
        if isinstance(x, list):
            #Strip spaces and convert to lowercase
            return [str.lower(i.replace(" ", "")) for i in x]
        else:
            #Check if director exists. If not, return empty string
            if isinstance(x, str):
                return str.lower(x.replace(" ", ""))
            else:
                return ''
                
    #Apply the generate_list function to cast, keywords, director and genres
    for feature in ['cast', 'director', 'genres', 'keywords']:
        df[feature] = df[feature].apply(sanitize)


    #Function that creates a soup out of the desired metadata
    def create_soup(x):
        return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

    # Create the new soup feature
    df['soup'] = df.apply(create_soup, axis=1)

    #Display the soup of the first movie
    #df.iloc[0]['soup']

    # Import CountVectorizer
    from sklearn.feature_extraction.text import CountVectorizer

    #Define a new CountVectorizer object and create vectors for the soup
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(df['soup'])

    #Import cosine_similarity function
    from sklearn.metrics.pairwise import cosine_similarity

    #Compute the cosine similarity score (equivalent to dot product for tf-idf vectors)
    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

    # Reset index of your df and construct reverse mapping again
    df = df.reset_index()
    indices2 = pd.Series(df.index, index=df['title'].str.lower())

    # Reset index of your df and construct reverse mapping again
    df = df.reset_index()
    indices2 = pd.Series(df.index, index=df['title'])

    return df, indices, cosine_sim, cosine_sim2
    #improvement, give more weight to director
    #improvement, include the rating
  
df, indices, cosine_sim, cosine_sim2 = func()  

df.to_csv('df.csv')
np.save('cosine_sim', cosine_sim)
np.save('cosine_sim2', cosine_sim2)
indices.to_csv('indices.csv')