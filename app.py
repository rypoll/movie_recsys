# Core Pkg
import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd
import numpy as np
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.metrics.pairwise import linear_kernel

@st.cache
def func():
    indices = pd.read_csv('indices.csv', header = 0, index_col = 0, squeeze = True)
    cosine_sim = np.load('cosine_sim.npy')
    cosine_sim2 = np.load('cosine_sim2.npy')
    df = pd.read_csv('df.csv')
    return df, indices, cosine_sim, cosine_sim2




df, indices, cosine_sim, cosine_sim2 = func()  
    
df_main = df.copy()
indices2 = indices.copy()
cosine_simc = cosine_sim.copy()
cosine_sim2c = cosine_sim2.copy()

@st.cache
def content_recommender_combined(title, cosine_sim, cosine_sim_meta, df, indices, num_of_rec=10):
    # Obtain the index of the movie that matches the title
    title = title.lower()
    if np.ndim(indices[title]) > 0:
        idx = indices[title].iloc[0]
    else: 
        idx = indices[title]

    # DESCRIPTION STUFF
    # Get the pairwsie similarity scores of all movies with that movie
    # And convert it into a list of tuples as described above
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    #create empty dataframe, this will be used for the combined dataframe
    df2 = pd.DataFrame()
    
    #set content score to the cosine sim of the content
    df2['content'] = cosine_sim[idx]
    #set meta score to the second cosine sim, the meta data stuff
    df2['meta'] = cosine_sim_meta[idx]  
    #incorporate the average rating into the algo
    df['vote_scaled'] = ((df['vote_average'])) * (df['vote_count']/df['vote_count'].max())
    df2['vote'] = df['vote_scaled']
    df2['vote'] =  df2['vote']/df2['vote'].max()
    #combine these scores, lets go with equal rating for now
    df2['combined_score'] = ((1.5*df2['content'] + 0.5*df2['meta'])/2)*(df2['vote']**0.1)
    #make the combined score to be a list
    comb_score = list(enumerate(df2['combined_score']))
    #sort the combined score
    comb_score = sorted(comb_score, key=lambda x: x[1], reverse=True)
    #get top 10 scores 
    comb_score = comb_score[1:]
    #get the indices for these movies
    movie_indices = [i[0] for i in comb_score]
    #return the top 10 similar movies
    final_recommended_movies =  df[['title', 'vote_average', 'year', 'overview']].iloc[movie_indices]
    return final_recommended_movies.head(num_of_rec)
    
  
	
	
	


	

	

	






@st.cache
def search_term_if_not_found(term,df):
    term = term.lower()
    result_df=df[df['title'].str.lower().str.contains(term).fillna(False)]
    return result_df[['title', 'year']]


def main():
    
    st.title("Movie recommendations based on a movie you like.")
    st.markdown("This is an app I created that uses [an IMDB movies dataset from Kaggle](https://www.kaggle.com/rounakbanik/the-movies-dataset) and allows the user to input a movie to find similar movies that the user might enjoy. Not all movies are present in this app due to lack of local computing power but feel free to try \"interstellar\" as a test!")

    
 
    
    
    menu = ["Recommendation", "Sample Data", "About" ]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Sample Data":
        st.subheader("Sample Data")
        st.dataframe(df_main.head(10))
        
    elif choice == "Recommendation":
        st.subheader("Recommend Movies:")
        search_term = st.text_input("What movie do you want to base your recommendations on?")
        num_of_rec = st.sidebar.number_input("Number of desired recommendations", 0,30, 5)
        if st.button("Recommend") or  search_term:
            if search_term is not None:
                try:    
                    result = content_recommender_combined(search_term, cosine_simc, cosine_sim2c, df_main, indices2, num_of_rec)
                    #for row in result.iterrows():
                    #    rec_title = row[1][0]
                    #    rec_score = row[1][1]
                    for line_number, (index,row) in enumerate(result.iterrows()):
                        rec_title = row[0]
                        rec_score = row[1]
                        rec_year = row[2]
                        rec_over = row[3]
                        line_num = str(line_number+1)
                        rec_score = str(rec_score)
                        rec_year = str(rec_year)
                        st.write(line_num, ".", rec_title, " (",rec_year, ")")
                        st.write("  ","Rating: ", rec_score)
                        st.write("  ", "Overview: ", rec_over)
                        st.write(" ")
                        #st.write( rec_title, rec_score
                                 
                                 
                    
                    #stc.html(RESULT_TEMP.format(rec_title,rec_score))
                    #stc.html(RESULT_TEMP.format(rec_title,rec_score),height=350)


                except:
                    result = "Movie not found. Did you mean the following movies?"
                    st.warning(result)
                    #st.info("Suggested Options include:")
                    result_df = search_term_if_not_found(search_term, df_main)
                    for line_number, (index,row) in enumerate(result_df.iterrows()):
                        rec_title = row[0]
                        rec_year = row[1]
                        line_num = str(line_number+1)
                        rec_year = str(rec_year)
                        st.write(line_num, ".", rec_title, " (",rec_year,")")
                        #st.write( rec_title, rec_score
                    #st.dataframe(result_df)
        st.subheader("How are movies recommended based off an inputted movie?")
        st.markdown("Each movie in the data contains a synopsis - a description of the plot of the movie. Movies also contain meta-data which captures things about the movie such as the director, the cast, and the genre of the movie. Using [Natural Language Processing](https://en.wikipedia.org/wiki/Natural_language_processing) and [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity), the synopsis and the meta-data for the inputted movie is compared the to synopses and meta-data of the other movies in the dataset. A score is given for each movie that indicates the degree of similarity between the movies, with higher scores being more similar. Factored into the similarity score is also the average rating and the number of votes of each of the movies - such that more popular and highly rated movies are more likely to appear in the recommendations. ")
        st.subheader("References:")
        st.markdown("* Banik, R., 2018. Hands-On Recommendation Systems with Python. Birmingham: Packt Publishing Ltd") 
                

    else: 
        st.subheader("About")
        st.text("Built with Streamlit")
    
if __name__ == '__main__':
    main()