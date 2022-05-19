import os
from numpy import positive
import pandas as pd 
import googleapiclient.discovery

def google_api(id):
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = "<Developer-Key-Here>"

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey = DEVELOPER_KEY)

    request = youtube.commentThreads().list(
        part="id,snippet",
        maxResults=100,
        order="relevance",
        videoId= id
    )
    response = request.execute()

    return response


video_ID = input("Video ID: ")
response = google_api(video_ID)


def create_df_author_comments():
  authorname = []
  comments = []
  for i in range(len(response["items"])):
    authorname.append(response["items"][i]["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"])
    comments.append(response["items"][i]["snippet"]["topLevelComment"]["snippet"]["textOriginal"])
  df_1 = pd.DataFrame(comments, index = authorname,columns=["Comments"])
  return df_1 

df = create_df_author_comments()



### Cleaning


import re
def cleaning_comments(comment):
  comment = re.sub("[🤣|🤭|🤣|😁|🤭|❤️|👍|🏴|😣|😠|💪|🙏]+",'',comment)
  comment = re.sub("[0-9]+","",comment)
  comment = re.sub("[\:|\@|\)|\*|\.|\$|\!|\?|\,|\%|\"]+"," ",comment)
  return comment
df["Comments"]= df["Comments"].apply(cleaning_comments)


def cleaning_comments1(comment):
  comment = re.sub("[💁|🌾|😎|♥|🤷‍♂]+","",comment)
  comment = re.sub("[\(|\-|\”|\“|\#|\!|\/|\«|\»|\&]+","",comment)
  return comment 

df["Comments"] = df["Comments"].apply(cleaning_comments1)


def cleaning_comments3(comment):
  comment = re.sub("\n"," ",comment)
  comment = re.sub('[\'|🇵🇰|\;|\！]+','',comment)
  return comment 
df["Comments"] = df["Comments"].apply(cleaning_comments3)

lower = lambda comment: comment.lower()
df['Comments'] = df['Comments'].apply(lower)

def remove_comments(df):
  # Checks for comments which has zero length in a dataframe
  zero_length_comments = df[df["Comments"].map(len) == 0]
  # taking all the indexes of the filtered comments in a list
  zero_length_comments_index = [ind for ind in zero_length_comments.index]
  # removing those rows from dataframe whose indexes matches 
  df.drop(zero_length_comments_index, inplace = True)
  return df
df = remove_comments(df)

from textblob.blob import TextBlob




def find_polarity_of_single_comment(text):
   return  TextBlob(text).sentiment.polarity

def find_polarity_of_every_comment(df):  
  df['Polarity'] = df['Comments'].apply(find_polarity_of_single_comment)
  return df 

df = find_polarity_of_every_comment(df)

analysis = lambda polarity: 'Positive' if polarity > 0 else 'Neutral' if polarity == 0 else 'Negative' 

def analysis_based_on_polarity(df):
  df['Analysis'] = df['Polarity'].apply(analysis)
  return df
  
df = analysis_based_on_polarity(df)

positive_comments = 0
negative_comments = 0
neutral_comments = 0

def print_positive_comments():
  sortedDF = df.sort_values(by=['Polarity']) 
  for i in range(0, sortedDF.shape[0] ):
    if( sortedDF['Analysis'][i] == 'Positive'):
      global positive_comments
      positive_comments += 1
print_positive_comments()

def print_negative_comments():
  sortedDF = df.sort_values(by=['Polarity']) 
  for i in range(0, sortedDF.shape[0] ):
    if( sortedDF['Analysis'][i] == 'Negative'):
      global negative_comments
      negative_comments += 1
print_negative_comments()


def print_neutral_comments():
  sortedDF = df.sort_values(by=['Polarity']) 
  for i in range(0, sortedDF.shape[0] ):
    if( sortedDF['Analysis'][i] == 'Neutral'):
      global neutral_comments
      neutral_comments += 1
print_neutral_comments()

Sum = positive_comments + negative_comments + neutral_comments
positive_comments = (positive_comments/Sum)*100

import matplotlib.pyplot as plt
from sklearn.feature_extraction import text 
import numpy as np

graph = np.array([positive_comments, negative_comments, neutral_comments])
labels = [f"Positive {positive_comments}%", f"Negative {negative_comments}%", f"Neutral {neutral_comments}%"]

plt.pie(graph, labels= labels)
plt.show()
