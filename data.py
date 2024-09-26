import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as pl
import nltk
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

df = pd.read_csv('Reviews.csv')

#print(df.info())
df['Summary'] = df['Summary'].astype(str)
pretext = df['Summary']
#plt = pl.bar(df, x='Score')
#plt.show()

text = " ".join(i for i in pretext)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="black").generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

