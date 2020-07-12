import pandas as pd
import os
import matplotlib.pyplot as plt
import re
from sklearn import feature_extraction, model_selection, naive_bayes, metrics
import numpy as np
from collections import Counter
from scipy.stats import binom

path = os.path.abspath(os.getcwd())
data_file = os.path.join(path, 'data.csv')  # women's clothing e-commerce reviews
data = pd.read_csv(data_file, sep=',', index_col=0)

#
# ----EDA-----
bin_rating = data['Rating'].value_counts()
plt.close('all')
bin_rating.plot(kind='bar')

bin_ratingType = data['Recommended IND'].value_counts()
plt.close('all')
bin_ratingType.plot(kind='pie')

bin_clothingType = data['Department Name'].value_counts()
plt.close('all')
bin_clothingType.plot(kind='bar')

bin_clothingType_detail = data['Class Name'].value_counts()
plt.close('all')
bin_clothingType_detail.plot(kind='bar')


# ---NB implementaiton from scratch ---
# -----Processing data-----------
def clean_text(sentence):
    sentence_clean = re.sub("[^a-zA-Z]", " ", sentence)
    return sentence_clean


def extract_words(sentence):
    words = sentence.split()
    ignore = ['and', 'the', 'is', 'and', 'it', 'this', 'to', 'in', 'on', 'for', 'of', 'with', 'was', 'my', 'that',
              'have', 'be', 'as', 'or', 'you', 'am', 'they']
    output = [w.lower() for w in words if w not in ignore and len(w) >= 2]
    return output


def get_words(df):
    text_wrap = {}
    words = {}
    words_list = {}
    n_words = {}
    for k in [0, 1]:
        text_wrap[k] = " ".join((df[df['Recommended IND'] == k][['title_reviews_combo']]).values.ravel())
        words[k] = extract_words(text_wrap[k])
        words_list_raw = Counter(words[k])
        words_list[k] = dict(sorted(words_list_raw.items(), key=lambda pair: pair[1], reverse=True))
        n_words[k] = len(words[k])
    return words_list, n_words


# combined title and reviews for text analysis
data["title_reviews_combo_raw"] = data["Title"].str.cat(data["Review Text"], sep="  ", na_rep="")
data["title_reviews_combo"] = data["title_reviews_combo_raw"].apply(lambda x: clean_text(x))

# clean text and initialize features
all_text = " ".join((data['title_reviews_combo']).values.ravel())
text2words = extract_words(all_text)
all_words_ = Counter(text2words)
all_words = dict(sorted(all_words_.items(), key=lambda pair: pair[1], reverse=True))
n_text_feature = len(all_words)

#
# splitting data into training data and testing data
def splitting(data, ratio):
    n_data = len(data)
    data_train = data.iloc[0:int(n_data * ratio)]
    data_test = data.iloc[(int(n_data * ratio) + 1):n_data]
    return data_train, data_test

data_train, data_test = splitting(data, 0.7)


#
# train model
def NB_train(data_train, a):
    words_train, n_words_train = get_words(data_train)
    p_feature_conditional = {}
    p_class = {}
    for k in range(0, 2):
        p_class[k] = len(data_train[data_train['Recommended IND'] == k]) / len(data_train)
        p_feature_conditional[k] = {h: 0 for h in all_words.keys()}
        for w in all_words.keys():
            if w in words_train[k]:
                p_feature_conditional[k][w] = (words_train[k][w] + a) / (n_words_train[
                                                                             k] + a * n_text_feature)  # calculate conditional probability with laplace smoothing
            else:
                p_feature_conditional[k][w] = a / (n_words_train[k] + n_text_feature)
    return p_class, p_feature_conditional

a = 1  # smoothing parameter
p_class, p_feature_conditional = NB_train(data_train, a)



# predict class
def calc_prob_binom(n, k, p):
    return binom.pmf(k, n, p)


def calc_prob_class(x_dict, n, p_cl, p_fc):
    y = {0: p_cl[0], 1: p_cl[1]}
    for w in x_dict.keys():
        for k in range(0, 2):
            y[k] = y[k] * calc_prob_binom(n, x_dict[w], p_fc[k][w])
    return y


# predict results for training data
def predict(new_data):
    new_data['predict'] = np.nan
    y_predict = {}
    for i, row in new_data.iterrows():
        line = extract_words(row['title_reviews_combo'])
        line_words_raw = Counter(line)
        n_line_words = len(line)
        line_words = dict(sorted(line_words_raw.items(), key=lambda pair: pair[1], reverse=True))
        class_prob = calc_prob_class(line_words, n_line_words, p_class, p_feature_conditional)
        # note that we did not calculate the actual probability of the class, i.e. p(K|X) = p(X|K) * p(K) / p(X)
        # we only calculates the nominator to select the most likely class, as p(X) is the same
        if class_prob[0] > class_prob[1]:
            y_predict[i] = 0
        else:
            y_predict[i] = 1
    new_data['predict'] = pd.Series(y_predict)
    return new_data

predict_class = predict(data_test.copy())

# evaluation results
def metrics(data_predict):
    match = data_predict[data_predict['Recommended IND'] == data_predict['predict']]
    recall = {}
    precision = {}
    F1={}
    w={}
    for k in range(0, 2):
        w[k]=np.sum(data_predict['Recommended IND'] == k)/len(data_predict)
        precision[k] = np.sum(match['predict'] == k) / np.sum(data_predict['Recommended IND'] == k)
        recall[k] = np.sum(match['predict'] == k) / np.sum(data_predict['predict'] == k)
        F1[k] = 2 * precision[k] * recall[k]/ (precision[k] + recall[k])
    return F1, precision, recall, w

def weighted_avg(x,w):
    ss=0
    for i in range(0,len(x)):
        ss=ss+x[i]*w[i]
    return ss

F1, precision, recall, w = metrics(predict_class)
df_F1 = pd.DataFrame(pd.Series(F1))
df_F1.plot(kind='bar')

# Calculate score based on macro average
recall_total = weighted_avg(recall, w)
precision_total = weighted_avg(precision, w)
F1_total = weighted_avg(F1, w)



## next step: introduce more features into the NB classifier, e.g. age, department


# ----use sk-learn-----
# ----generating features and modelling via Naive Bayes-----
def analyze(stop_words_):
    if stop_words_ == "":
        vectorizer = feature_extraction.text.CountVectorizer(stop_words=None)
    else:
        vectorizer = feature_extraction.text.CountVectorizer(stop_words=stop_words_)

    X = vectorizer.fit_transform(data['title_reviews_combo'])

    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, data['Recommended IND'])

    NB = naive_bayes.MultinomialNB()
    NB.fit(x_train, y_train)

    y_predict = NB.predict(x_test)
    unique, counts = np.unique(y_predict, return_counts=True)
    dict(zip(unique, counts))

    score_train = NB.score(x_train, y_train)
    score_test = NB.score(x_test, y_test)
    recall_test = metrics.recall_score(y_test, NB.predict(x_test))
    precision_test = metrics.precision_score(y_test, NB.predict(x_test))

    return score_train, score_test, recall_test, precision_test


score_train, score_test, recall_test, precision_test = analyze("english")
