import nltk
import tensorflow, numpy, tensorflow, random, json
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

#Load Json data
with open("data.json") as doc:
  data = json.load(doc)

words = []
labels = []
docs = []

#Get pattern of the words
for sentence in data["sentences"]:
  for pattern in sentence["patterns"]:
    wrds = nltk.word_tokenize(pattern)#Sort through words by pattern
    words.extend(wrds)
    docs.append(pattern)

  if sentence["tag"] not in labels:
    labels.append(sentence["tag"])
