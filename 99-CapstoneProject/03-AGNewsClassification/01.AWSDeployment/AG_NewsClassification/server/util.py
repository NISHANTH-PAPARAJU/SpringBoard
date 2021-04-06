import pickle
import json
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, WordPunctTokenizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import word2vec
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words = stopwords.words('english')


__stop_words = stopwords.words('english')
__w2v_vectorizer = None
__LogReg_w2v_model = None
__labels = None
__predicted_category = None
__confidence = None
__class_confidences = None


def run_article_prediction(headline, short_desc):
    global __predicted_category
    global __class_confidences
    global __confidence

    __predicted_category, predicted_probs = predict_news_type(headline, short_desc, __w2v_vectorizer, __LogReg_w2v_model,
                                                              embedType='w2v', labels=__labels, lemmatize=True)

    __class_confidences = predicted_probs[0]
    __confidence = round(max(__class_confidences) * 100, 2)


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __w2v_vectorizer
    global __LogReg_w2v_model
    global __labels

    if __labels is None:
        with open("./artifacts/labels.json", 'r') as f:
            __labels = json.load(f)['labels']
        # with open("./artifacts/columns.json", "r") as f:
        #     __data_columns = json.load(f)['data_columns']
        #     __locations = __data_columns[3:]  # first 3 columns are sqft, bath, bhk

    if __w2v_vectorizer is None:
        with open("./artifacts/w2v.pkl", "rb") as f:
            __w2v_vectorizer = pickle.load(f)

    if __LogReg_w2v_model is None:
        with open("./artifacts/LogReg_w2v.pkl", "rb") as f:
            __LogReg_w2v_model = pickle.load(f)

    print("loading saved artifacts...done")


def get_predicted_cat():
    return __predicted_category


def get_confidence_score():
    return __confidence


def get_class_confidences():
    return __class_confidences

# def get_location_names():
#     return __locations


# def get_data_columns():
#     return __data_columns

def clean_corpus(corpus, tocase='lower', remove_punc=True, punctuations=list(string.punctuation), remove_whitespace=True,
                 stopwords=__stop_words, remove_numbers=True, remove_urls=True, lemmatize=False):
    """
    Takes the corpus as input and performs the corpus cleaning as required,
    then returns the detokenized corpus.
    """

    cleaned_corpus = corpus
    # Tokenize
    tokens = word_tokenize(corpus)

    # Convert Multi Lingual Text
    # will be done later

    # Convert the corpus to one case (lower or Upper)
    valid_tokens = [token.lower() for token in tokens]

    # Remove Punctuations
    if remove_punc:
        valid_tokens = [
            token for token in valid_tokens if token not in punctuations]

    # Remove White Space
        # will be done later

    # Remove Other Special Characters.
    if remove_numbers:
        valid_tokens = [token for token in valid_tokens if re.search(
            '[0-9]+', token) is None]

    # Remove urls.
    if remove_urls:
        valid_tokens = [token for token in valid_tokens if re.search(
            'https+|http+', token) is None]

    # Remove stop words
    valid_tokens = [token for token in valid_tokens if token not in stopwords]

    # lemmatization / stemming
    if lemmatize:
        word_lem = WordNetLemmatizer()
        valid_tokens = [word_lem.lemmatize(token) for token in valid_tokens]
    else:
        pst = PorterStemmer()
        valid_tokens = [pst.stem(token) for token in valid_tokens]

    # De-tokenize
    cleaned_corpus = "".join([" "+i if not i.startswith("'") and i not in '!%\'()*+,-./:;<=>?@[\\]^_`{|}~'
                              else i for i in valid_tokens]).strip()
    return cleaned_corpus


def get_aggreagated_embedding(doc, model):
    """
    Takes the normalized-tokenized corpus as input and
    returns the average embeddings of document usingnp the model. (Averages the word embeddings)
    """
    vocab = model.wv.vocab.keys()
    doc_embed = np.average([model.wv[word.lower()]
                            for word in doc if word.lower() in vocab], axis=0)
    return doc_embed


def document_vectorizer(corpus, w2v_model, doc_action='clean_tokenize', lemmatize=False):
    """
    Takes the list of documents (corpus), w2v_model as input
    Returns the document embeddings averaged on each word.

    Attributes:
      doc_action:  possible values ('clean_tokenize', 'tokenize', 'no_action')
                   Default : clean_tokenize
    """
    document_embeddings = []
    doc_actions = ('clean_tokenize', 'tokenize', 'no_action')
    if not doc_action in doc_actions:
        raise NameError("Invalid parameter passed for doc_action attribute.")
    for doc in corpus:
        norm_doc = []
        if doc_action.lower() == 'clean_tokenize':
            clean_doc = clean_corpus(doc, lemmatize=lemmatize)
            wpt = WordPunctTokenizer()
            norm_doc = wpt.tokenize(clean_doc)
        elif doc_action.lower() == 'tokenize':
            wpt = WordPunctTokenizer()
            norm_doc = wpt.tokenize(doc)
        elif doc_action.lower() == 'no_action':
            norm_doc = doc
        else:
            raise NameError(
                "Invalid parameter passed for doc_action attribute.")
        doc_embed = get_aggreagated_embedding(norm_doc, w2v_model)
        document_embeddings.append(doc_embed)
    return np.array(document_embeddings)


def predict_news_type(headline, short_desc, vec, model, embedType='w2v', labels=["World", "Sports", "Business", "Sci/Tech"], lemmatize=True):
    """
    Takes the headline, short description, vectorizer, model and embedding type as input.
    Then predicts the type of the news article among ['World', 'Sports', 'Business', 'Sci/Tech'].
    """
    clubbed_article = headline + " " + short_desc
    clean_clubbed_article = clean_corpus(clubbed_article)
    if embedType.lower() == 'tfidf':
        article_embed = vec.transform(clean_clubbed_article)
    elif embedType.lower() == 'w2v':
        clubbed_article_list = []
        clubbed_article_list.append(clean_clubbed_article)
        article_embed = document_vectorizer(
            clubbed_article_list, vec, doc_action='tokenize', lemmatize=lemmatize)
    else:
        raise NameError("Invalid 'embedType' specified.")
    y_pred = model.predict_proba(article_embed.reshape(1, -1))
    pos = np.argmax(y_pred)
    label = labels[pos]
    # print("Article Type: " + label + ", Confidence: " +
    #       str(round((y_pred[0][pos])*100, 2)) + "%")
    return label, y_pred


if __name__ == '__main__':
    load_saved_artifacts()
    # print(get_estimated_price('1st Phase JP Nagar', 1000, 3, 3))
    # print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    # print(get_estimated_price('Kalhalli', 1000, 2, 2))  # other location
    # print(get_estimated_price('Ejipura', 1000, 2, 2))  # other location
