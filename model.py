from arrow import FORMAT_ATOM
import numpy as np
import pandas as pd
import sklearn

movies = pd.read_csv('C:/Users/Thansheer/OneDrive/Desktop/Machine learning project/Movie recommendation system/tmdb_5000_movies.csv')
credits = pd.read_csv('C:/Users/Thansheer/OneDrive/Desktop/Machine learning project/Movie recommendation system/tmdb_5000_credits.csv')
movies=movies.merge(credits,on='title')
movies['original_language'].value_counts()
movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]
movies.isnull().sum()
movies.dropna(inplace=True)
movies.duplicated().sum()
movies.iloc[0].genres
#to take the names 
#to convert the strings to list
import ast 
def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L
#to convert the  to list
movies['genres']=movies['genres'].apply(convert)
movies['keywords']=movies['keywords'].apply(convert)
movies.head()
movies['keywords'].values
#to select top 3 cast
def convert3(obj):
    L=[]
    counter= 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter +=1 
        else:
            break
    return L
#code
movies['cast']=movies['cast'].apply(convert3)
movies.head
movies['crew'][0]
#to select the director
def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L
movies['crew']=movies['crew'].apply(fetch_director)
#to make overview into list
movies['overview']=movies['overview'].apply(lambda x:x.split())

#to remove the space bt two words
movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])
movies.head(1)
movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']
new_df = movies[['movie_id','title','tags']]
#now convert tags to a string
new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))
new_df['tags'][0]
new_df['tags']=new_df['tags'].apply(lambda x:x.lower())
new_df.head(1).values
#to convert the text into a vector
import sklearn.feature_extraction.text
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
vectors=cv.fit_transform(new_df['tags']).toarray()
#to check the repeated words
cv.get_feature_names_out()

#to appply steming (to make similar words into one word on the new_df)
import nltk



import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()
def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


#code for main df
new_df['tags']=new_df['tags'].apply(stem)

#to convert the text into a vector(2 after steming)
import sklearn.feature_extraction.text
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
vectors=cv.fit_transform(new_df['tags']).toarray()
#to check the repeated words
cv.get_feature_names_out()

#to find the distance (similarity) bt the vectors
from sklearn.metrics.pairwise import cosine_similarity
similarity=cosine_similarity(vectors)

#to sort the movies in basis of similarity bt each outher movie and sort wrt similarity not the number

sorted(list(enumerate(similarity[0])),reverse=True, key=lambda x:x[1])[1:6]

#remond system
def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distances =similarity[movie_index]
    movie_list =sorted(list(enumerate(distances)),reverse=True, key=lambda x:x[1])[1:6]
    for i in movie_list:
        i=i[0]
        print(new_df.iloc[i].title)
        

recommend('Titanic')




