import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as pl
import nltk
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('Reviews.csv')

#print(df.info())
df['Summary'] = df['Summary'].astype(str)
pretext = df['Summary']
score = df['Score']
#plt = pl.bar(df, x='Score')
#plt.show()

text = " ".join(i for i in pretext)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="black").generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
#plt.show()

#Sentiment
df = df[df['Score'] != 3]
df['sentiment'] = df['Score'].apply(lambda x: 1 if x > 3 else -1)

#print(df.head())

#positive
filtered_df = df[(df['sentiment'] == 1)]
text = " ".join(filtered_df['Summary'])
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="black").generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
#plt.show()

#negative
filtered_df = df[(df['sentiment'] == -1)]
text = " ".join(filtered_df['Summary'])
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="black").generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
#plt.show()

#plt = pl.bar(df, x='sentiment')
#plt.show()

pattern = r'[^\w\s]'
df['Summary'] = df['Summary'].replace(pattern, '', regex=True)

division = df[['Summary','sentiment']]
#print(division.head())

vectorizer = CountVectorizer()
vectorizer.fit(df[['Summary','sentiment']])

print("Vocabulary: ", vectorizer.vocabulary_)
vector = vectorizer.transform(df[['Summary','sentiment']])
 
print("Encoded Document is:")
print(vector.toarray())









