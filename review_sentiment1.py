import pandas as pd
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from progress.bar import Bar
import spacy

print("Packages Loaded\n")

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return(text)

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
def sentiment_sen(x):
    if x >= 0.00 : 
        return("Positive") 
  
    elif x <= - 0.10 : 
        return("Negative") 
  
    else : 
        return("Neutral") 
    
reviews = pd.read_csv("metacritic_critic_reviews.csv")

print("##################################START##################################")
print("\nFormat Cells")

rev = reviews['review']

print("\nNormalise Reviews")
bar = Bar('Processing', fill = "|", max=len(reviews), suffix='%(percent).1f%%')
for i in range(len(reviews)):
    reviews.loc[i,'n_review'] = clean_text(reviews.loc[i,'review'])
    bar.next()
bar.finish()

print("\nSentiment Analysis")
sid = SentimentIntensityAnalyzer()
reviews["sentiments"] = reviews["n_review"].apply(lambda x: sid.polarity_scores(x))
reviews = pd.concat([reviews.drop(['sentiments'], axis=1), reviews['sentiments'].apply(pd.Series)], axis=1)

reviews["sentiment"] = reviews["compound"].apply(lambda x: sentiment_sen(x))

print("##################################ASPECT BASED SENTIMENT ANALYSIS##################################")

def find_sentiment(text):
    text = clean_text(text)
    sid = SentimentIntensityAnalyzer()
    sentiments = sid.polarity_scores(text)
    return sentiment_sen(sentiments['compound'])

aspects = []
sentences = reviews['review'].to_list()
print("Aspect Sentiments")
for sentence in sentences:
  doc = nlp(sentence)
  descriptive_term = ''
  target = ''
  for token in doc:
    if token.dep_ == 'nsubj' and token.pos_ == 'NOUN':
      target = token.text
    if token.pos_ == 'ADJ':
      prepend = ''
      for child in token.children:
        if child.pos_ != 'ADV':
          continue
        prepend += child.text + ' '
      descriptive_term = prepend + token.text  
      aspects.append({'aspect': target,
    'description': find_sentiment(descriptive_term)})
      
reviews['aspect_sentiment'] = pd.Series(aspects)

reviews.to_csv("tweets_sentiment.csv")
print("Reviews with overall sentiment analysis and aspect based sentiment analysis saved as 'reviews_sentiment.csv'")
print("##################################FINISH##################################")