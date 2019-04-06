from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from os import listdir
from sklearn.feature_extraction.text import CountVectorizer as vectorizer

word_data = "The best performance can bring in sky high success."

folder = "aclImdb/"
test_folder = "test/"
train_folder = "train/"
neg_folder = "neg/"
pos_folder = "pos/"
print("Getting Contents From Training Files")
path = folder + train_folder + pos_folder
pro_train  = [open(path + f, 'r').read() for f in listdir(path)]
path = folder + train_folder + neg_folder
neg_train  = [open(path + f, 'r').read() for f in listdir(path)]

print("Getting Contents From Test Files")
path = folder + test_folder + pos_folder
profile = listdir(path)
pro_test  = [open(path + f, 'r').read() for f in listdir(path)]
path = folder + test_folder + neg_folder
negfile = listdir(path)
neg_test  = [open(path + f, 'r').read() for f in listdir(path)]


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
print(X_Train.shape)
print("Making Counts for Test Data")
X_test = vect.transform(pro_test + neg_test)
classifiers = {#'random_forest': RandomForestClassifier(),
               #'bayes' : MultinomialNB(),
               'sgd': SGDClassifier(loss='hinge')}
wrong_pos = []
wrong_neg = []
for name, model in classifiers.items():
    print("Training " + name)
    model.fit(X, y)
    predictions = model.predict(X_test)
    print(name, f1_score(predictions, y))
    pos = 0 
    pos_cont = []
    for i in range(12500):
      if predictions[i] != y[i]:
        pos += 1
        pos_cont += [profile[i]]
    neg = 0
    neg_cont = []
    for i in range(12500, 25000):
      if predictions[i] != y[i]:
        neg += 1
        neg_cont += [negfile[i-13000]]

print(pos_cont[0:3])
print(len(pos_cont))
print(neg_cont[-3:])
print(len(neg_cont))
"""
val = f1_score(predictions, y)
print(val)

print(arr)
print(arr.shape)
print(type(X))
print(vect.vocabulary_)
"""

