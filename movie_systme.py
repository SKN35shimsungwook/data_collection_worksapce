
import numpy as np
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt
import seaborn as sns

#파일 로딩
def data_load():
  movies = pd.read_csv('./raw/m1/movies.dat', delimiter='::', header=None, engine='python', encoding='ISO-8859-1',
                    names=['MovieID', 'Title', 'Genres'] )
  users = pd.read_csv('./raw/m1/users.dat', sep='::', engine='python',header=None,
                    names=['UserID','Gender','Age','Occupation','Zip-code'])
  ratings = pd.read_csv('./raw/m1/ratings.dat', sep='::', engine='python', header=None,
                    names=['UserID','MovieID','Rating','Timestamp'])
  return movies, users, ratings

# merge
def data_merge(movies, users, ratings):
  data = ratings.merge(users).merge(movies)
  recommendation_data = data[['UserID', 'MovieID', 'Rating']]
  return recommendation_data

#  pivot, corr
def data_pivot_corr(recommendation_data):
  recommendation_pivot = recommendation_data.pivot(index ='UserID', columns='MovieID', values='Rating')
  recommendation_pivot.fillna(0, inplace=True)
  return recommendation_pivot

def nearest_user(small_test_corr, user_id, n):
  return small_test_corr.loc[user_id].sort_values(ascending =False)[1: n+1]

def movie_seen(user_id):
   return recommendation_pivot.loc[user_id][recommendation_pivot.loc[user_id]>0]

def recommend_movie(recommendation_pivot, movies, user_id, n):
  small_test_corr = recommendation_pivot.T.iloc[:500,:500].corr()
  user_list = nearest_user(small_test_corr, user_id, n).index
  user_mv_list = recommendation_data[(recommendation_data.UserID.isin(user_list)) & (recommendation_data.Rating==5)]
  user7_mv_list = movie_seen(user_id)
  unseen_list = set(user_mv_list['MovieID']) - set(user7_mv_list.index)
  return movies[movies['MovieID'].isin(unseen_list)].reset_index(drop=True)

if __name__  == '__main__':
  movies, users, ratings = data_load()
  recommendation_data = data_merge(movies, users, ratings)
  recommendation_pivot = data_pivot_corr(recommendation_data)
  userid =  int(input('UserId 입력:'))
  movie_receive = recommend_movie(recommendation_pivot, movies,  userid, 2)
  print(len(movie_receive))
  print(movie_receive)
