
# Text clean and pre-processing
import contractions
import string
import nltk
import re

from bs4 import BeautifulSoup
from abbreviations import abbreList
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from num2words import num2words

nltk.download(['punkt', 'stopwords', 'wordnet'])

def cleanAndPreprocess(text, isCapitalization = True, isExpandConstract = True,
 isConvertAbbre = True, isConverNumber = True, isStemming = True, isLemmatize = True ):
    # Noise removal: remove html, xml tag,...
        # Remove html tag
    text =  removeHTMLtags(text)
       
    # Capitalization
    if isCapitalization:
        text = text.lower()

    # Expand contractions
    if isExpandConstract:
        text = expandContractions(text)

    # Slangs and Abbreviations
    if isConvertAbbre:
        text = convertAbbre(text)

    # Noise removal:
     # Remove punctuation
    text = re.sub(r'[^\w\s]',"", text)

    # Tokenization
    text = word_tokenize(text)
    
    # Remove stop words
    text = removeStopWords(text)

    # Convert number to word
    if isConverNumber:
        text = convertNumber(text)

    # Correct Spelling

    # Stemming
    if isStemming:
        text = stemmingText(text)

    # Lemmatization
    if isLemmatize:
        text = lemmatizeText(text)

    return text

# Expand contractions function
def expandContractions(text):
    return contractions.fix(text)

def convertAbbre(text):
    _text = []
    for word in text.split():
        _text.append(abbreList[word] if word in abbreList else word)
    return (" ").join(_text)

# Remove stop words
def removeStopWords(text):
    stopWords = set(stopwords.words('english'))
    filter = []
    for word in text:
        if word.lower() not in stopWords:
            filter.append(word)
    return filter

# Convert number to text
def convertNumber(text):
    _text = []
    filter = []
    for word in text:
        if word.lower().isdigit():
            filter.append(num2words(word))
        else:
            filter.append(word)
    return filter

# Stemming function
def stemmingText(text):
    porter = PorterStemmer()
    return [porter.stem(word.lower()) for word in text]

# Lemmatizing text
def lemmatizeText(text):
    wordNet = WordNetLemmatizer()
    return [wordNet.lemmatize(word.lower()) for word in text]

def removeHTMLtags(text):
    soup = BeautifulSoup(text, "html.parser")
    _text = soup.get_text(separator=" ")
    return _text