import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from datetime import datetime
import plotly.express as px
from plotly.offline import plot
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings("ignore")
#%%getCredentials
localPath = '/Users/ilketopak/Documents/GitHub/ilkot/'
f = open(localPath+"creds.json")
creds = json.load(f)
cid = creds["cid"]
secret = creds["secret"]

#%%connection
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
#%%
def getSongsFromPlaylist(playlistId,sp):

    testPlaylist = sp.playlist(playlistId)
    #fetch data in lists
    artistName = []
    artistId = []
    trackName = []
    popularity = []
    trackId = []
    duration = []
    addedDate=[]
    for i,t in enumerate(testPlaylist['tracks']['items']):
        addedDate.append(t['added_at'])
        artistName.append(t['track']['artists'][0]['name'])
        artistId.append(t['track']['artists'][0]['id'])
        trackName.append(t['track']['name'])
        trackId.append(t['track']['id'])
        popularity.append(t['track']['popularity'])
        duration.append(t['track']['duration_ms'])
    
    #create df format
    playlistDf = pd.DataFrame({'artistId':artistId,
                              'artists':artistName,
                              'trackId':trackId,
                              'name':trackName,
                              'duration_ms':duration,
                              'addedDate':addedDate})
    
    return artistId, trackId, playlistDf

def getAudioFeatures(trackId,sp):
    
    #we can get all track if list supplied in
    trackAudioFeats = sp.audio_features(trackId)
    
    audioFeatureDf = pd.DataFrame()
    for i,t in enumerate(trackAudioFeats):
        tempAudioDf = pd.DataFrame(t,index=[i]) 
        audioFeatureDf = audioFeatureDf.append(tempAudioDf)
    
    audioDropCols = ['type','uri','track_href','analysis_url','duration_ms','time_signature']
    audioFeatureDf.drop(audioDropCols,axis=1,inplace=True)
    
    return audioFeatureDf

def getTrackFeatures(trackId):
    trackIdPop = []
    pop = []
    explicit = []
    releaseDate =[]
    for t in trackId:
        trackIdPop.append(t)
        trackDict = sp.track(t)
        pop.append(trackDict['popularity'])
        explicit.append(trackDict['explicit'])
        releaseDate.append(trackDict["album"]["release_date"])
        
    trackDf = pd.DataFrame({"id":trackIdPop,"trackPopularity":pop,"explicit":explicit,"release_date":releaseDate})
    return trackDf

def listChunks(mylist, chunk_size):
    return [mylist[offs:offs+chunk_size] for offs in range(0, len(mylist), chunk_size)]

def getArtistFeatures(artistId):
    artistIdSingle = list(dict.fromkeys(artistId))
    #spotify lets only 50 artist in a single query, so split it to max 50 items
    artistIdChunks = listChunks(artistIdSingle,50)
    artistFeats = list()
    for a in artistIdChunks:
        xx = sp.artists(a)['artists']
        artistFeats.append(xx)
    
    #combine lists
    artistFeats = sum(artistFeats,[])
    
    artistIds = []
    artistPops = []
    artistFollowers = []
    artistGenre = []
    for t in artistFeats:
        artistFollowers.append(t["followers"]["total"])
        artistPops.append(t["popularity"])
        artistIds.append(t["id"])
        try:
            artistGenre.append(t["genres"])
        except IndexError:
            artistGenre.append(list())
            
            
    artistDf = pd.DataFrame({"artistId":artistIds,
                             "artistPopularity":artistPops,
                             "artistFollowers":artistFollowers,
                             "artistGenre":artistGenre})
    return artistDf





def orderPlaylist(playlistDf):
    #rename columns
    newColOrder = ['acousticness','artists','danceability','duration_ms','energy',
                   'explicit','trackId','instrumentalness','key','liveness','loudness',
                   'mode','name','trackPopularity','release_date','speechiness','tempo',
                   'valence','artistGenre','addedDate']
    
    playlistDfNew = playlistDf[newColOrder]
    
    playlistDfNew.rename(columns = {"trackId":"id","trackPopularity":"popularity","artistGenre":"consolidates_genre_lists","addedDate":"date_added"},inplace=True)
    playlistDfNew['year'] = playlistDfNew['release_date'].apply(lambda x: x.split('-')[0])
    playlistDfNew['popularity_red'] = playlistDfNew['popularity'].apply(lambda x: int(x/5))
    playlistDfNew['date_added'] = pd.to_datetime(playlistDfNew['date_added'])
    
    return playlistDfNew



def getPlaylistDf(playlistId,sp):
    artistId, trackId, playlistDf = getSongsFromPlaylist(playlistId,sp)
    audioFeatureDf = getAudioFeatures(trackId, sp)
    trackDf= getTrackFeatures(trackId)
    artistDf = getArtistFeatures(artistId)

    #merge with main df
    playlistDf = pd.merge(playlistDf,audioFeatureDf,how='left',left_on ='trackId',right_on='id').drop('id',axis=1).drop_duplicates()
    #merge with playlistDf
    playlistDf = pd.merge(playlistDf,trackDf,how='left',left_on='trackId',right_on='id').drop('id',axis=1).drop_duplicates()
    #merge with playlistDf
    playlistDf = pd.merge(playlistDf,artistDf,on="artistId")
    
    playlistDf = orderPlaylist(playlistDf)
    
    return playlistDf


#%%test
playlistId = '041ef61Og4SZJdhTauuJol'
playlistDf = getPlaylistDf(playlistId,sp)




#%%



