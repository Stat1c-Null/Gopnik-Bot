import nltk
import tensorflow, numpy, tensorflow, random, json
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

#Load Json data
with open("data.json") as document:
  data = json.load(document)

words = []
labels = []
docs_one = []
docs_two = []

#Get pattern of the words
for sentence in data["sentences"]:
  for pattern in sentence["patterns"]:
    wrds = nltk.word_tokenize(pattern)#Sort through words by pattern
    words.extend(wrds)
    docs_one.append(pattern)
    docs_two.append(sentence["tag"])

  if sentence["tag"] not in labels:
    labels.append(sentence["tag"])

#Push words together and remove any duplicate words to create a vocabulary
words = [stemmer.stem(w.lower()) for w in words]    
words = sorted(list(set(words)))#Sort through vocabulary

labels = sorted(labels)

#Neural Networks don't understand strings, so we will convert strings into list of numbers that show how frequently words appear in the sentence
train = [] #Bags of words
result = []
empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_one):
  bag = []

  wrds = [stemmer.stem(w) for w in doc]

  #Check if the word exists, and if so put 1 for it in the array
  for w in words:
    if w in wrds:
      bag.append(1)
    else:#If it doesnt put 0
      bag.append(0)

  result_row = empty[:]
  result_row[labels.index(docs_two[x])] = 1

  train.append(bag)
  result.append(result_row)

train = numpy.array(train)
result = numpy.array(result)