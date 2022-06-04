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
#%%getCredentials
localPath = '/Users/ilketopak/Documents/GitHub/ilkot/spotifyAnalysisAndRecom/'
f = open(localPath+"creds.json")
creds = json.load(f)
cid = creds["cid"]
secret = creds["secret"]

#%%connection
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

#%%getTracksFromPlaylist
playlistId = '4yS19wMR2YuxGwGdukwigU'
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
                          'artistName':artistName,
                          'trackId':trackId,
                          'trackName':trackName,
                          'duration':duration,
                          'addedDate':addedDate})

#duration conversion to minutes
playlistDf["duration"] = playlistDf["duration"]/60000 

#datetime conversion
date_format = "%Y-%m-%dT%H:%M:%SZ"
playlistDf["dateTime"] = playlistDf["addedDate"].apply(lambda x: datetime.strptime(x,date_format))
playlistDf["date"] = playlistDf["dateTime"].apply(lambda x: x.date())
playlistDf["hour"] = playlistDf["dateTime"].apply(lambda x: x.time().hour)

#%%audio features
#we can get all track if list supplied in
trackAudioFeats = sp.audio_features(trackId)

audioFeatureDf = pd.DataFrame()
for i,t in enumerate(trackAudioFeats):
    tempAudioDf = pd.DataFrame(t,index=[i]) 
    audioFeatureDf = audioFeatureDf.append(tempAudioDf)

audioDropCols = ['type','uri','track_href','analysis_url','duration_ms']
audioFeatureDf.drop(audioDropCols,axis=1,inplace=True)

#merge with main df
playlistDf = pd.merge(playlistDf,audioFeatureDf,how='left',left_on ='trackId',right_on='id').drop('id',axis=1).drop_duplicates()

#create replacements for enumerations on db
modeDict = {0:"minor",1:"major"}
keyDict = {0:"C",1:"C#",2:"D",
           3:"D#",4:"E",5:"F",
           6:"F#",7:"G",8:"G#",
           9:"A",10:"A#",11:"B"}

playlistDf["mode"].replace(modeDict,inplace=True)
playlistDf["key"].replace(keyDict,inplace=True)

#%%track features
trackIdPop = []
pop = []
for t in trackId:
    pop.append(sp.track(t)['popularity'])
    trackIdPop.append(t)
    
trackDf = pd.DataFrame({"id":trackIdPop,"trackPopularity":pop})
#merge with playlistDf
playlistDf = pd.merge(playlistDf,trackDf,how='left',left_on='trackId',right_on='id').drop('id',axis=1).drop_duplicates()


#%%artistFeatures
#remove duplicate artists
artistIdSingle = list(dict.fromkeys(artistId))

def listChunks(mylist, chunk_size):
    return [mylist[offs:offs+chunk_size] for offs in range(0, len(mylist), chunk_size)]

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
artistGenre1 = []
artistGenre2 = []
for t in artistFeats:
    artistFollowers.append(t["followers"]["total"])
    artistPops.append(t["popularity"])
    artistIds.append(t["id"])
    try:
        artistGenre1.append(t["genres"][0])
    except IndexError:
        artistGenre1.append("noGenre")
    try:
        artistGenre2.append(t["genres"][1])
    except IndexError:
        artistGenre2.append("noGenre")
        
        
artistDf = pd.DataFrame({"artistId":artistIds,
                         "artistPopularity":artistPops,
                         "artistFollowers":artistFollowers,
                         "artistGenre1":artistGenre1,
                         "artistGenre2":artistGenre2})
 
#merge with playlistDf
playlistDf = pd.merge(playlistDf,artistDf,on="artistId").drop_duplicates()

#%%visualizations
playlistDf.columns
#counts
artistCounts = playlistDf["artistName"].value_counts()[:5].reset_index()
artistCounts.columns = ['artistName','count']
keyCounts = playlistDf["key"].value_counts().reset_index() ## add other numerical metrics, for C what is the average danceability
keyCounts.columns = ['key','count']
genreCounts = playlistDf["artistGenre1"].value_counts()[:10].reset_index()
genreCounts.columns = ['genre','count']
modeCounts = playlistDf["mode"].value_counts().reset_index()
modeCounts.columns = ['mode','count']

#countplots
#first 5 artists
sns.barplot(data=artistCounts,y='count', x='artistName')
plt.xticks(rotation=90)
#which keys
sns.barplot(data=keyCounts,y='count',x='key')
plt.show()
#which genres
sns.barplot(data=genreCounts,y='count',x='genre')
plt.xticks(rotation=90)
#major or minor
sns.barplot(data = modeCounts, y='count',x='mode')


#distrubitions
numCols = ['duration','danceability','energy',
           'loudness','acousticness',
           'liveness','valence',
           'tempo','trackPopularity','artistPopularity',
           'mode','artistGenre1','key']
numPlayDf = playlistDf[numCols]
# numerical distrubitions
numPlayDf.hist(layout=(6,2),figsize=(20, 30))

#pairplot by different categories
sns.pairplot(numPlayDf,hue='mode')
sns.pairplot(numPlayDf,hue='key')
sns.pairplot(numPlayDf,hue='genre')


#%%subplots for distrubitions
from plotly.subplots import make_subplots
rows=2
cols=5
fig = make_subplots(rows=rows, cols=cols, subplot_titles=numCols)  
x, y = np.meshgrid(np.arange(rows)+1, np.arange(cols)+1)
count  = 0
for row, col in zip(x.T.reshape(-1), y.T.reshape(-1)):
    fig.add_trace(
        go.Histogram(x = numPlayDf[numCols[count]].values),
        row = row,
        col = col
    )
    count+=1
    
fig.update_layout(height=900, width=1500, title_text='Feature Distribution', showlegend=False)
fig.show()
plot(fig)

#%%box plot 
figBox = px.box(data_frame=numPlayDf,y='duration',color='key')
figBox.update_layout(height=900, width=1500, title_text='BoxPlot Duration', showlegend=True)
plot(figBox)

#%%
#pivots
playlistPvt = playlistDf.describe().T.reset_index()[['index','mean']]
colNames=list(playlistPvt['index'])
playlistPvt.drop(['index'],axis=1,inplace=True)
playlistPvt = playlistPvt.T
playlistPvt.columns = colNames
selCols = ['danceability',
           'energy',
           'instrumentalness',
           'liveness',
           'valence']
playlistPvt = playlistPvt[selCols].T.reset_index()
playlistPvt.columns = ['metric','average']



sns.barplot(data=playlistPvt,x="metric",y='average')
plt.ylim(0,1)

pxBar = px.bar(playlistPvt,x='metric',y='average')
plot(pxBar,filename='spotiyTest.html')


artistPvt = pd.pivot_table(playlistDf, index='artistName',
                           values=['danceability','energy','loudness','instrumentalness','liveness','valence','tempo','trackPopularity','artistPopularity','artistFollowers']      
                           ,aggfunc={'danceability':np.mean,
                                     'energy':np.mean,
                                     'loudness':np.mean,
                                     'instrumentalness':np.mean,
                                     'liveness':np.mean,
                                     'valence':np.mean,
                                     'tempo':np.mean,
                                     'trackPopularity':np.mean,
                                     'artistPopularity':np.max,
                                     'artistFollowers':np.max})


#%%
