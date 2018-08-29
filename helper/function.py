import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF as NMF_sklearn

from datetime import *



def add_reviews_cnt(listings, reviews):
    last30 = listings.last_scraped[0] - timedelta(30)
    last60 = listings.last_scraped[0] - timedelta(60)
    last90 = listings.last_scraped[0] - timedelta(90)
    last120 = listings.last_scraped[0] - timedelta(120)
    last180 = listings.last_scraped[0] - timedelta(180)
    
    reviewslast30days = reviews[reviews.date>last30].groupby('listing_id').count()
    reviewslast60days = reviews[reviews.date>last60].groupby('listing_id').count()
    reviewslast90days = reviews[reviews.date>last90].groupby('listing_id').count()
    reviewslast120days = reviews[reviews.date>last120].groupby('listing_id').count()
    reviewslast180days = reviews[reviews.date>last180].groupby('listing_id').count()
    
    reviews_count = pd.concat([reviewslast30days,reviewslast60days,
                               reviewslast90days,reviewslast120days,
                              reviewslast180days],axis=1).fillna(0)
    reviews_count.columns=['last30','last60','last90','last120','last180']
    listings = pd.merge(listings, reviews_count,how='left', left_on='id', right_index=True)
    return listings


def select_active_host(df_listings, nb_availability, nb_reviews, my_condition):
    mask_check_list = (df_listings['availability_30']==nb_availability)
    listings_to_check = df_listings[mask_check_list]
    listings_left = df_listings[~mask_check_list]
    
    print('{} of listings to check'.format(listings_to_check.shape[0]))
    
    condition = listings_to_check[my_condition]>=nb_reviews
    listings_to_keep = listings_to_check[condition]
    
    lost = listings_to_check.shape[0] - listings_to_keep.shape[0]
    pct = lost / listings_to_check.shape[0] * 100
    print('Remove {} listings. {:.2f}% of listings checked.  \n'.format(lost, pct))
    return pd.concat([listings_left,listings_to_keep])

def add_holidays(x, df_holidays):
    t1 = x 
    t2 = x + timedelta(30)
    for idx in range(df_holidays.shape[0]):
        if df_holidays.iloc[idx,0] >= t1 and df_holidays.iloc[idx,0]<t2:
            return df_holidays.iloc[idx,1]
    
    return 'No_holiday'


class PricePerBedroom:
    def __init__(self):
        self.mean_train_df_with_month = None
        self.mean_train_df_without_month = None

    def fit(self, df):
        self.mean_train_df_with_month = df.groupby(['neighbourhood_cleansed','month_scraped'])[['price_per_bedroom']].mean()
        self.mean_train_df_without_month = df.groupby('neighbourhood_cleansed')[['price_per_bedroom']].mean()

    def transform(self, df):
        df = df.join(self.mean_train_df_with_month, on=['neighbourhood_cleansed', 'month_scraped'],
                      how='left', rsuffix='_per_neig_month')
        
        df['diff_price_per_bedroom_hood_month'] = df['price_per_bedroom'] - df['price_per_bedroom_per_neig_month']
        df['diff_price_per_bedroom_hood_month'] = df['diff_price_per_bedroom_hood_month'].fillna(df['diff_price_per_bedroom_hood_month'].mean())

        df = df.join(self.mean_train_df_without_month, on='neighbourhood_cleansed',
                      how='left', rsuffix='_per_neighbourhood')
        df['diff_price_per_bedroom_hood'] = df['price_per_bedroom'] - df['price_per_bedroom_per_neighbourhood']
        df['diff_price_per_bedroom_hood'] = df['diff_price_per_bedroom_hood'].fillna(df['diff_price_per_bedroom_hood'].mean())
        
        df.drop('price_per_bedroom_per_neig_month', axis=1, inplace=True)
        df.drop('price_per_bedroom_per_neighbourhood', axis=1, inplace=True)
        return df

def drop_columns(df, list_to_drop):
    df_new = df.copy()
    for name in list_to_drop:
        if name in df_new.columns:
            df_new.drop(name, axis=1,inplace=True)

    return df_new




class text:
    def __init__(self):
        self.W_train = None
        self.H_train = None
        self.vectorizer = None
        self.vocabulary = None
        self.nmf = None
        self.prob_w = None
        self.hand_labels = None

    def softmax(self, v, temperature=1.0):
        '''
        A heuristic to convert arbitrary positive values into probabilities.
        See: https://en.wikipedia.org/wiki/Softmax_function
        '''
        expv = np.exp(v / temperature)
        s = np.sum(expv, axis=1)
        return expv / s[:,None]

    def build_text_vectorizer(self, contents, use_tfidf=True, use_stemmer=False, max_features=None):
        '''
        Build and return a **callable** for transforming text documents to vectors,
        as well as a vocabulary to map document-vector indices to words from the
        corpus. The vectorizer will be trained from the text documents in the
        `contents` argument. If `use_tfidf` is True, then the vectorizer will use
        the Tf-Idf algorithm, otherwise a Bag-of-Words vectorizer will be used.
        The text will be tokenized by words, and each word will be stemmed if
        `use_stemmer` is True. If `max_features` is not None, then the vocabulary
        will be limited to the `max_features` most common words in the corpus.
        '''
        Vectorizer = TfidfVectorizer if use_tfidf else CountVectorizer
        tokenizer = RegexpTokenizer(r"[\w']+")
        stem = PorterStemmer().stem if use_stemmer else (lambda x: x)
        stop_set = set(stopwords.words('english'))

        # Closure over the tokenizer et al.
        def tokenize(text):
            tokens = tokenizer.tokenize(text)
            stems = [stem(token) for token in tokens if token not in stop_set]
            return stems

        vectorizer_model = Vectorizer(tokenizer=tokenize, max_features=1000,ngram_range=(1,1))
        vectorizer_model.fit(contents)
        vocabulary = np.array(vectorizer_model.get_feature_names())

        # Closure over the vectorizer_model's transform method.
        def vectorizer(X):
            return vectorizer_model.transform(X).toarray()

        return vectorizer, vocabulary

    def hand_label_topics(self, H, vocabulary, prob_w, series):
        '''
        Print the most influential words of each latent topic, and prompt the user
        to label each topic. The user should use their humanness to figure out what
        each latent topic is capturing.
        '''
        hand_labels = []
        for i, row in enumerate(H):
            idx = np.argsort(prob_w[:,i])[::-1][:1]
            for k in idx:
                print('Example: {}'.format(series[k]))
            print()
            top_five = np.argsort(row)[::-1][:20]
            print('topic', i)
            print('-->', ' '.join(vocabulary[top_five]))
            label =  str(series.name) + '_' + str(input('please label this topic: '))
            hand_labels.append(label)
            print()
        return hand_labels

    def train_text_features(self, series, n_components):
        series = series.fillna('unknown')
        vectorizer, vocabulary = self.build_text_vectorizer(series,
                                    use_tfidf=True,
                                    use_stemmer=True,
                                    max_features=None)
        self.vectorizer = vectorizer
        self.vocabulary = vocabulary
        X = vectorizer(series)
        nmf = NMF_sklearn(n_components=n_components, max_iter=100, alpha=0.0)
        nmf_model = nmf.fit(X)
        self.nmf = nmf_model
        W = nmf.transform(X)
        H = nmf.components_
        self.W_train = W
        self.H_train = H
        prob_w = self.softmax(W, temperature=0.01)
        self.prob_w
        hand_labels = self.hand_label_topics(H, vocabulary, prob_w, series)
        self.hand_labels = hand_labels
        return pd.DataFrame(W,columns=[hand_labels])

    def predict_text_features(self, series):
        series = series.fillna('unknown')
        X = self.vectorizer(series)
        W = self.nmf.transform(X)
        H = self.nmf.components_
        self.W_test = W
        self.H_test = H
        return pd.DataFrame(W, columns=[self.hand_labels])

def my_train_test_split(df, date):
    mask_test = df.last_scraped >= pd.to_datetime(date)
    X_train = df[~mask_test]
    X_test = df[mask_test]

    y_train = X_train.pop('availability_30')
    y_train = y_train.values
    y_test = X_test.pop('availability_30')
    y_test = y_test.values

    return X_train, X_test, y_train, y_test

def precleaning(X):
    df = X.copy()

    # clean price
    df.price=df.price.str.replace(r'[$,]','').astype(float)

    # clean zipcode
    df.zipcode = pd.to_numeric(df.zipcode,errors='coerce')
    df = df[df.zipcode.notnull()]
    df.zipcode = df.zipcode.astype(int)
    df.zipcode = df.zipcode.astype(str)
    mask = df.zipcode.isin(['94306','94015','60614','94080','94066','94129','94607','94005','94106','95014'])
    df = df[~mask]

    if 'cleaning_fee' in df.columns:
        df.cleaning_fee=df.cleaning_fee.str.replace(r'[$,]','').astype(float)

    if 'extra_people' in df.columns:
        df.extra_people=df.extra_people.str.replace(r'[$,]','').astype(float)

    # Convert 0 bed and 0 bedrooms to 1.
    if 'bedrooms' in df.columns:
        df.loc[df['bedrooms']==0,'bedrooms'] = 1
        df.bedrooms = df.bedrooms.fillna(1)
    if 'beds' in df.columns:
        df.loc[df['beds']==0,'beds'] = 1
        df.beds = df.beds.fillna(1)

    # Convert minor cases for property type to 'other property types'
    if 'property_type' in df.columns:
        mask = df.property_type.isin(['Condominium','Condominium','Guest suite','Townhouse'])
        df.loc[mask,'property_type'] = 'Apartment'
        mask = df.property_type.isin(['Apartment','House'])
        df.loc[~mask, 'property_type'] = 'Others'


    # transform t/f column to 1 and 0
    for columns_t_f in ['host_is_superhost','instant_bookable','host_identity_verified']:
        if columns_t_f in df.columns:
            df[columns_t_f] = df[columns_t_f].replace({'f':0,'t':1})

    # deal with nan
    if 'cleaning_fee' in df.columns:
        df.cleaning_fee = df.cleaning_fee.fillna(0)
    if 'reviews_per_month' in df.columns:
        df.reviews_per_month = df.reviews_per_month.fillna(0)
    if 'review_scores_rating' in df.columns:
        df.review_scores_rating = df.review_scores_rating.fillna(0)
    if 'host_response_time' in df.columns:
        df.host_response_time = df.host_response_time.fillna('unclear')
    if 'host_is_superhost' in df.columns:
        df.host_is_superhost = df.host_is_superhost.fillna(0)
    if 'host_identity_verified' in df.columns:
        df.host_identity_verified = df.host_identity_verified.fillna(0)


    # df.to_csv('data/listings/listings.csv')

    return df

