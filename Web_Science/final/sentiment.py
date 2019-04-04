from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import nltk
from os import listdir
from sklearn.feature_extraction.text import CountVectorizer as vectorizer

word_data = "The best performance can bring in sky high success."
nltk_tokens = nltk.word_tokenize(word_data)  	

folder = "aclImdb/"
test_folder = "test/"
train_folder = "train/"
neg_folder = "neg/"
pos_folder = "pos/"
print("Getting Contents From Training Files")
path = folder + train_folder + pos_folder
onlyfiles = listdir(path)
pro_train  = [open(path + f, 'r').read() for f in listdir(path)]
path = folder + train_folder + neg_folder
neg_train  = [open(path + f, 'r').read() for f in listdir(path)]

print("Getting Contents From Test Files")
path = folder + test_folder + pos_folder
onlyfiles = listdir(path)
pro_test  = [open(path + f, 'r').read() for f in listdir(path)]
path = folder + test_folder + neg_folder
neg_test  = [open(path + f, 'r').read() for f in listdir(path)]


def bigram(text_file):
  text = open(text_file, 'r').read()
  tokens = nltk.word_tokenize(text)
  return list(nltk.bigrams(tokens))


vect = vectorizer()

#vect.set_params(tokenizer=tokenizer.tokenize)

# remove English stop words
vect.set_params(stop_words='english')
# include 1-grams and 2-grams
print("Making vocabulary")
vect.set_params(ngram_range=(2, 2))
X = vect.fit_transform(pro_train + neg_train)
print("Vocabulary Made")
X_Train = (X.toarray())
y = [1] * len(pro_train) + [0] * len(neg_train)
print("Making Counts for Test Data")
X_test = vect.transform(pro_test + neg_test)
bayes = MultinomialNB()
print("Fitting Bayes model")
bayes.fit(X, y)
print("Making Predictions for Test Model")
predictions = bayes.predict(X_test)
val = precision_score(predictions, y, average='micro')
print(val)

"""
print(arr)
print(arr.shape)
print(type(X))
print(vect.vocabulary_)
"""


