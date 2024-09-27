import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as pl
import nltk
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('Reviews.csv')

print(df.info())
df['Summary'] = df['Summary'].astype(str)
pretext = df['Summary']
score = df['Score']
plt = pl.bar(df, x='Score')
plt.show()

text = " ".join(i for i in pretext)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="black").generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#Sentiment
df = df[df['Score'] != 3]
df['sentiment'] = df['Score'].apply(lambda x: 1 if x > 3 else -1)
print(df.head())

#positive
filtered_df = df[(df['sentiment'] == 1)]
text = " ".join(filtered_df['Summary'])
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="black").generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#negative
filtered_df = df[(df['sentiment'] == -1)]
text = " ".join(filtered_df['Summary'])
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="black").generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

plt = pl.bar(df, x='sentiment')
plt.show()

pattern = r'[^\w\s]'
df['Summary'] = df['Summary'].replace(pattern, '', regex=True)

division = df[['Summary','sentiment']]
print(division.head())


print(len(division))
split_index = int(len(division) * 0.8)
df_80 = df.iloc[:split_index]
df_20 = df.iloc[split_index:]

print("Training DataFrame (80%):")
print(len(df_80))
print("\nTesting DataFrame (20%):")
print(len(df_20))

vectorizer = CountVectorizer()

bag_of_words = vectorizer.fit_transform(df_80)
