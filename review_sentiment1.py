import pandas as pd # Pandas is used for dataset operation
import string # String is used for identifying parts of grammar; in the program's case it is punctations 
from nltk import pos_tag # Identifies the Parts-of-Speach tags of the word, like, "Canada" is noun
from nltk.corpus import stopwords # A list of words that don't add a significant meaning to the statement during processing 
from nltk.stem import WordNetLemmatizer # Lemmatization is a process which converts words in multiple forms to a easy to 
                                        # process word, like, "runs" and "running" turns to "run" 
from nltk.corpus import wordnet 
from nltk.sentiment.vader import SentimentIntensityAnalyzer # The Sentiment Analysis model which is trained and is readily available by the modulw developers
from progress.bar import Bar # Adding a progress bar to keep track of the progress of the functions
import spacy # A superior form of the NLTK package which offers better analytical tools for examining the relationships between the words

print("Packages Loaded\n")

# spaCy offers the computations operating on the GPU, which during development showed the total time taken to process the data significantly reduce
# If the user doesn't have a GPU, please comment the statement below
spacy.prefer_gpu() 
nlp = spacy.load("en_core_web_sm") # A pretrained module which can tokenize the statements

# The function below cleans the text, so that the Sentiment Analysis model can process the information effictively
def clean_text(text):
    # Lower text
    text = text.lower()
    # Tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # Remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # Remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # Remove empty tokens
    text = [t for t in text if len(t) > 0]
    # POS tag text
    pos_tags = pos_tag(text)
    # Lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # Remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # Join all
    text = " ".join(text)
    return(text)

# The function below works with the clean_text function which will convert the POS tag from a string form to the wordnet format
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

# This function will convert the compound socre generated by the Sentiment Analyser model to a easy to understand label
# More labels can be added based on the user's preference
# The limits set are arbitrary and can be modified based on the user's perference     
def sentiment_sen(x):
    if x >= 0.00 : 
        return("Positive") 
  
    elif x <= - 0.10 : 
        return("Negative") 
  
    else : 
        return("Neutral") 

# Load the dataset; dataset must be a CSV form and the user must change the file name to read any other file
# The statements that are to be processed must belong in the column 'review'
reviews = pd.read_csv("metacritic_critic_reviews.csv") 

print("##################################START##################################")
print("\nFormat Cells")

rev = reviews['review'] # Select the review column only

print("\nNormalise Reviews")
bar = Bar('Processing', fill = "|", max=len(reviews), suffix='%(percent).1f%%') # Set up a progress bar to track the progress
for i in range(len(reviews)):
    reviews.loc[i,'n_review'] = clean_text(reviews.loc[i,'review']) # Each statement is cleaned and tokenized and are then ammended to a new column in the dataset
    bar.next()
bar.finish()

print("\nSentiment Analysis")
sid = SentimentIntensityAnalyzer()
# All clean/normalized text are provided as input to the model and the polarity socres are generated
reviews["sentiments"] = reviews["n_review"].apply(lambda x: sid.polarity_scores(x)) 
reviews = pd.concat([reviews.drop(['sentiments'], axis=1), reviews['sentiments'].apply(pd.Series)], axis=1) # Adjustments are made to make the dataset more readable

reviews["sentiment"] = reviews["compound"].apply(lambda x: sentiment_sen(x)) # The socres generated are converted to lables

print("##################################ASPECT BASED SENTIMENT ANALYSIS##################################")

# The following function finds sentiments, similar to the code segment above, but are only applied for words or small sets of words to understand it's sentiments
def find_sentiment(text):
    text = clean_text(text)
    sid = SentimentIntensityAnalyzer()
    sentiments = sid.polarity_scores(text)
    return sentiment_sen(sentiments['compound'])

aspects = [] # Aspects are the parts of the statement that the sentiments are reffering
sentences = reviews['review'].to_list()
print("Aspect Sentiments")
# Following couple lines of code are for the progress bar to keep track of the progress
length = len(sentences)
bar = Bar('Processing', fill = "|", max=length, suffix='%(percent).1f%%')
# Each review is analyzed to identify the aspects, the words that provide its sentiments and the sentiments towards the aspects
for i in range(length):
    sentence = sentences[i]
    doc = nlp(sentence)
    descriptive_term = ''
    target = ''
    for token in doc:
        # All sentiment delivering words are Adjectives and it describes Nouns
        # The following lines of code look at the words in the statement and if they are nouns that have links (i.e 'nsubj') then they are the aspects
        # And if the words are adjectives and are not linked to any adverbs then they are the descriptive terms 
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
            'description': descriptive_term, 
        'sentiment': find_sentiment(descriptive_term)}) # Adds the infromation in a dictionary format contining the aspects, description and sentimetnts

    bar.next()
bar.finish
      
reviews['aspect_sentiment'] = pd.Series(aspects) # Converts to a pandas seeries and adds it to the final resulting dataset

reviews.to_csv("review_sentiment.csv") # Saves the resulting dataset as a CSV file
print("Reviews with overall sentiment analysis and aspect based sentiment analysis saved as a .csv file")
print("##################################FINISH##################################")