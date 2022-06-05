import pandas as pd
import numpy as np
import json
import re 
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import spotipy.util as util
import getPlaylist
import warnings
warnings.filterwarnings("ignore")

#%%getCredentials
localPath = '/Users/ilketopak/Documents/GitHub/ilkot/spotifyAnalysisAndRecom/'
f = open(localPath+"script/spotyRecom/creds.json")
creds = json.load(f)
cid = creds["cid"]
secret = creds["secret"]
token = creds["token"]

#%%
def ohe_prep(df, column, new_name): 
    """ 
    Create One Hot Encoded features of a specific column

    Parameters: 
        df (pandas dataframe): Spotify Dataframe
        column (str): Column to be processed
        new_name (str): new column name to be used
        
    Returns: 
        tf_df: One hot encoded features 
    """
    
    tf_df = pd.get_dummies(df[column])
    feature_names = tf_df.columns
    tf_df.columns = [new_name + "|" + str(i) for i in feature_names]
    tf_df.reset_index(drop = True, inplace = True)    
    return tf_df

def create_feature_set(df, float_cols):
    """ 
    Process spotify df to create a final set of features that will be used to generate recommendations

    Parameters: 
        df (pandas dataframe): Spotify Dataframe
        float_cols (list(str)): List of float columns that will be scaled 
        
    Returns: 
        final: final set of features 
    """
    
    #tfidf genre lists
    tfidf = TfidfVectorizer()
    tfidf_matrix =  tfidf.fit_transform(df['consolidates_genre_lists'].apply(lambda x: " ".join(x)))
    genre_df = pd.DataFrame(tfidf_matrix.toarray())
    genre_df.columns = ['genre' + "|" + i for i in tfidf.get_feature_names()]
    genre_df.reset_index(drop = True, inplace=True)

    #explicity_ohe = ohe_prep(df, 'explicit','exp')    
    year_ohe = ohe_prep(df, 'year','year') * 0.5
    popularity_ohe = ohe_prep(df, 'popularity_red','pop') * 0.15

    #scale float columns
    floats = df[float_cols].reset_index(drop = True)
    scaler = MinMaxScaler()
    floats_scaled = pd.DataFrame(scaler.fit_transform(floats), columns = floats.columns) * 0.2

    #concanenate all features
    final = pd.concat([genre_df, floats_scaled, popularity_ohe, year_ohe], axis = 1)
     
    #add song id
    final['id']=df['id'].values
    
    return final


def create_necessary_outputs(playlist_name,id_dic, df):
    """ 
    Pull songs from a specific playlist.

    Parameters: 
        playlist_name (str): name of the playlist you'd like to pull from the spotify API
        id_dic (dic): dictionary that maps playlist_name to playlist_id
        df (pandas dataframe): spotify datafram
        
    Returns: 
        playlist: all songs in the playlist THAT ARE AVAILABLE IN THE KAGGLE DATASET
    """
    
    #generate playlist dataframe
    playlist = pd.DataFrame()
    playlist_name = playlist_name

    for ix, i in enumerate(sp.playlist(id_dic[playlist_name])['tracks']['items']):
        #print(i['track']['artists'][0]['name'])
        playlist.loc[ix, 'artist'] = i['track']['artists'][0]['name']
        playlist.loc[ix, 'name'] = i['track']['name']
        playlist.loc[ix, 'id'] = i['track']['id'] # ['uri'].split(':')[2]
        playlist.loc[ix, 'url'] = i['track']['album']['images'][1]['url']
        playlist.loc[ix, 'date_added'] = i['added_at']

    playlist['date_added'] = pd.to_datetime(playlist['date_added'])  
    
    playlist = playlist[playlist['id'].isin(df['id'].values)].sort_values('date_added',ascending = False)
    
    return playlist

def generate_playlist_feature(complete_feature_set, playlist_df, weight_factor):
    """ 
    Summarize a user's playlist into a single vector

    Parameters: 
        complete_feature_set (pandas dataframe): Dataframe which includes all of the features for the spotify songs
        playlist_df (pandas dataframe): playlist dataframe
        weight_factor (float): float value that represents the recency bias. The larger the recency bias, the most priority recent songs get. Value should be close to 1. 
        
    Returns: 
        playlist_feature_set_weighted_final (pandas series): single feature that summarizes the playlist
        complete_feature_set_nonplaylist (pandas dataframe): 
    """
    #check if the songs is in complete feature set basically do they exist in rawData and select existed songs
    complete_feature_set_playlist = complete_feature_set[complete_feature_set['id'].isin(playlist_df['id'].values)]#.drop('id', axis = 1).mean(axis =0)
    #merge with date added-we'll use it as a weight controller
    complete_feature_set_playlist = complete_feature_set_playlist.merge(playlist_df[['id','date_added']], on = 'id', how = 'inner')
    #exclude the songs in selected playlist from main table(complete feature set or rawData)
    complete_feature_set_nonplaylist = complete_feature_set[~complete_feature_set['id'].isin(playlist_df['id'].values)]#.drop('id', axis = 1)
    #sort values
    playlist_feature_set = complete_feature_set_playlist.sort_values('date_added',ascending=False)
    #find most recent date_added in playlist
    most_recent_date = playlist_feature_set.iloc[0,-1]
    
    #create month interval between the most recent date added and other added dates
    for ix, row in playlist_feature_set.iterrows():
        playlist_feature_set.loc[ix,'months_from_recent'] = int((most_recent_date.to_pydatetime() - row.iloc[-1].to_pydatetime()).days / 30)
    
    #add weight based on months from recent, we would like to have similar songs that we added recently
    playlist_feature_set['weight'] = playlist_feature_set['months_from_recent'].apply(lambda x: weight_factor ** (-x))
    
    playlist_feature_set_weighted = playlist_feature_set.copy()
    #multiply derived weights with other features to diminish their equal weights that calculated before
    playlist_feature_set_weighted.update(playlist_feature_set_weighted.iloc[:,:-4].mul(playlist_feature_set_weighted.weight,0))
    #select only feature set columns
    playlist_feature_set_weighted_final = playlist_feature_set_weighted.iloc[:, :-4]
    
    return playlist_feature_set_weighted_final.sum(axis = 0), complete_feature_set_nonplaylist

def generatePlaylistFeature(complete_feature_set, playlistDf, weight_factor):
    
    playlistFloatCols = playlistDf.dtypes[playlistDf.dtypes == 'float64'].index.values
    
    playlist_feature_set = create_feature_set(playlistDf,float_cols=playlistFloatCols)
    colsCompleteFeatureSet = complete_feature_set.head(0)
    playlist_feature_set = pd.concat([colsCompleteFeatureSet,playlist_feature_set],axis=0,ignore_index=True).fillna(0)
    playlist_feature_set = playlist_feature_set[colsCompleteFeatureSet.columns]
    playlist_feature_set = playlist_feature_set.merge(playlistDf[['id','date_added']], on='id',how='inner')
    playlist_feature_set = playlist_feature_set.sort_values('date_added',ascending=False)
    
    complete_feature_set_nonplaylist = complete_feature_set[~complete_feature_set['id'].isin(playlistDf['id'].values)]
    #find most recent date_added in playlist
    most_recent_date = playlist_feature_set.iloc[0,-1]
    
    for ix, row in playlist_feature_set.iterrows():
        playlist_feature_set.loc[ix,'months_from_recent'] = int((most_recent_date.to_pydatetime() - row.iloc[-1].to_pydatetime()).days / 30)
    
    playlist_feature_set['weight'] = playlist_feature_set['months_from_recent'].apply(lambda x: weight_factor ** (-x))
    playlist_feature_set_weighted = playlist_feature_set.copy()
    playlist_feature_set_weighted.update(playlist_feature_set_weighted.iloc[:,:-4].mul(playlist_feature_set_weighted.weight,0))
    playlist_feature_set_weighted_final = playlist_feature_set_weighted.iloc[:, :-4]
    playlist_feature_set_weighted_final = playlist_feature_set_weighted_final.sum(axis = 0)
    
    return playlist_feature_set_weighted_final, complete_feature_set_nonplaylist

    
def generate_playlist_recos(df, features, nonplaylist_features):
    """ 
    Pull songs from a specific playlist.

    Parameters: 
        df (pandas dataframe): spotify dataframe
        features (pandas series): summarized playlist feature
        nonplaylist_features (pandas dataframe): feature set of songs that are not in the selected playlist
        
    Returns: 
        non_playlist_df_top_40: Top 40 recommendations for that playlist
    """
    
    non_playlist_df = df[df['id'].isin(nonplaylist_features['id'].values)]
    non_playlist_df['sim'] = cosine_similarity(nonplaylist_features.drop('id', axis = 1).values, features.values.reshape(1, -1))[:,0]
    non_playlist_df_top_10 = non_playlist_df.sort_values('sim',ascending = False).head(10)
    #non_playlist_df_top_10['url'] = non_playlist_df_top_10['id'].apply(lambda x: sp.track(x)['album']['images'][1]['url'])
    
    return non_playlist_df_top_10


localPath = '/Users/ilketopak/Documents/GitHub/ilkot/spotifyAnalysisAndRecom/'
#%%getData
rawData = pd.read_csv(localPath+"/data/data.csv")
dataWGenre = pd.read_csv(localPath+"/data/data_w_genres.csv")

#%%data cleaning&manipulation
#convert genres to a list since genres columns is string
dataWGenre['genres_upd'] = dataWGenre['genres'].apply(lambda x: [re.sub(' ','_',i) for i in re.findall(r"'([^']*)'", x)])

#convert artist to list from strings
rawData['artists_upd_v1'] = rawData['artists'].apply(lambda x: re.findall(r"'([^']*)'", x))

#artists with aposthrophe in their names like Lovin' McLaren
rawData['artists_upd_v2'] = rawData['artists'].apply(lambda x: re.findall('\"(.*?)\"',x))

#combine those two columns
rawData['artists_upd'] = np.where(rawData['artists_upd_v1'].apply(lambda x: not x), rawData['artists_upd_v2'], rawData['artists_upd_v1'] )

#create new column combining artist name and song name since same song name can be from different artists
#so with drop duplicates we need to prevent this situation
rawData['artists_song'] = rawData.apply(lambda row: row['artists_upd'][0]+row['name'],axis = 1)
rawData.sort_values(['artists_song','release_date'], ascending = False, inplace = True)

#drop duplicates
rawData.drop_duplicates('artists_song',inplace = True)

#%%create artist genres for each id and combine them into one list for each id
artists_exploded = rawData[['artists_upd','id']].explode('artists_upd')
artists_exploded_enriched = artists_exploded.merge(dataWGenre, how = 'left', left_on = 'artists_upd',right_on = 'artists')
artists_exploded_enriched_nonnull = artists_exploded_enriched[~artists_exploded_enriched.genres_upd.isnull()]

#combine genres by id
artists_genres_consolidated = artists_exploded_enriched_nonnull.groupby('id')['genres_upd'].apply(list).reset_index()

#combine them in one list
artists_genres_consolidated['consolidates_genre_lists'] = artists_genres_consolidated['genres_upd'].apply(lambda x: list(set(list(itertools.chain.from_iterable(x)))))

#merge it with rawData
rawData = rawData.merge(artists_genres_consolidated[['id','consolidates_genre_lists']], on = 'id',how = 'left')
#%%create features
#split year
rawData['year'] = rawData['release_date'].apply(lambda x: x.split('-')[0])

#select float columns
float_cols = rawData.dtypes[rawData.dtypes == 'float64'].index.values

# create 5 point buckets for popularity 
rawData['popularity_red'] = rawData['popularity'].apply(lambda x: int(x/5))

# tfidf can't handle nulls so fill any null values with an empty list
# if d is not a list replace it with empty list
rawData['consolidates_genre_lists'] = rawData['consolidates_genre_lists'].apply(lambda d: d if isinstance(d, list) else [])

#select only one artist no need to featured artists
rawData['artists'] = rawData["artists_upd"].iloc[0][0]


#%%

complete_feature_set = create_feature_set(rawData, float_cols=float_cols)#.mean(axis = 0)


#%%get personal spotify playlists
#getCredentials
localPath = '/Users/ilketopak/Documents/GitHub/ilkot/'
f = open(localPath+"creds.json")
creds = json.load(f)
cid = creds["cid"]
secret = creds["secret"]

scope = "user-library-read"
token = util.prompt_for_user_token(scope, client_id= cid, client_secret=secret, redirect_uri='http://localhost:8882/callback')
sp = spotipy.Spotify(auth=token)

playlistId = '0ZqGujD1DW2ByD5toZuwlN'
playlistDf = getPlaylist.getPlaylistDf(playlistId, sp)
playlistDf['date_added'] = pd.to_datetime(playlistDf['date_added'])
playlistDf.info()

complete_feature_set_playlist_vector_1, complete_feature_set_nonplaylist_1 = generatePlaylistFeature(complete_feature_set, playlistDf, 1.09)

#%%




#showerTop10 = generate_playlist_recos(rawData, complete_feature_set_playlist_vector_Shower, complete_feature_set_nonplaylist_Shower)

reco10 = generate_playlist_recos(rawData,complete_feature_set_playlist_vector_1,complete_feature_set_nonplaylist_1)


#%%
https://open.spotify.com/playlist/0ZqGujD1DW2ByD5toZuwlN?si=dc8606a5d71e40ad
https://open.spotify.com/playlist/4r2gDE3SxQEtEy8h0MpnMQ?si=bc6821c1338a4bee
https://open.spotify.com/playlist/6SNiMQ0XTwV7J5PAilHgJI?si=b9a815535c944775