import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import warnings
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
import requests
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load training and testing datasets
train = pd.read_csv('train.csv')
train_original = train.copy()

test = pd.read_csv('test.csv')
test_original = test.copy()

# Combine train and test datasets for consistent preprocessing
combined_data = pd.concat([train, test], ignore_index=True, sort=True)

# Function to remove specific patterns (e.g., Twitter handles) from text
def remove_pattern(text, pattern):
    matches = re.findall(pattern, text)
    for match in matches:
        text = re.sub(match, "", text)
    return text

# Clean tweets by removing Twitter handles and filtering short words
combined_data['Cleaned_Tweets'] = combined_data['tweet'].apply(lambda x: remove_pattern(str(x), r"@\w*"))
combined_data['Cleaned_Tweets'] = combined_data['Cleaned_Tweets'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))

# Tokenization and stemming using Porter Stemmer
ps = nltk.PorterStemmer()
combined_data['Clean_Tweets'] = combined_data['Cleaned_Tweets'].apply(lambda x: ' '.join([ps.stem(w) for w in x.split()]))

# Load word cloud mask image
mask_url = 'http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png'
mask_image = np.array(Image.open(requests.get(mask_url, stream=True).raw))
image_color = ImageColorGenerator(mask_image)

# Function to generate and display WordCloud
def generate_wordcloud(text, title, interpolation):
    wc = WordCloud(background_color='black', height=1500, width=4000, mask=mask_image).generate(text)
    plt.figure(figsize=(10, 20))
    plt.imshow(wc.recolor(color_func=image_color), interpolation=interpolation)
    plt.axis('off')
    plt.title(title, fontsize=20)
    plt.show()

# Generate WordClouds for positive and negative tweets
positive_words = ' '.join(combined_data['Clean_Tweets'][combined_data['label'] == 0])
negative_words = ' '.join(combined_data['Clean_Tweets'][combined_data['label'] == 1])

generate_wordcloud(positive_words, 'Positive Tweets WordCloud', 'hamming')
generate_wordcloud(negative_words, 'Negative Tweets WordCloud', 'gaussian')

# Function to extract hashtags from tweets
def extract_hashtags(tweets):
    return sum([re.findall(r'#(\w+)', tweet) for tweet in tweets], [])

# Extract hashtags for positive and negative tweets
positive_hashtags = extract_hashtags(combined_data['Clean_Tweets'][combined_data['label'] == 0])
negative_hashtags = extract_hashtags(combined_data['Clean_Tweets'][combined_data['label'] == 1])

# Function to plot top hashtags
def plot_top_hashtags(hashtags, title):
    freq_dist = nltk.FreqDist(hashtags)
    df = pd.DataFrame({'Hashtags': list(freq_dist.keys()), 'Count': list(freq_dist.values())})
    top_hashtags = df.nlargest(20, 'Count')
    sns.barplot(data=top_hashtags, y='Hashtags', x='Count')
    plt.title(title)
    plt.xlabel('Count')
    plt.ylabel('Hashtags')
    plt.xticks(rotation=45)
    sns.despine()
    plt.show()

# Plot top hashtags for positive and negative tweets
plot_top_hashtags(positive_hashtags, 'Top Positive Hashtags')
plot_top_hashtags(negative_hashtags, 'Top Negative Hashtags')

# Feature extraction using Bag of Words (BoW) and TF-IDF
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words="english")
bow = bow_vectorizer.fit_transform(combined_data['Clean_Tweets'])

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(combined_data['Clean_Tweets'])

# Split the data into training and testing sets
train_size = train.shape[0]
train_bow, train_tfidf = bow[:train_size], tfidf[:train_size]

# Perform train-test split
x_train_bow, x_valid_bow, y_train_bow, y_valid_bow = train_test_split(train_bow, train['label'], test_size=0.3, random_state=2)
x_train_tfidf, x_valid_tfidf, y_train_tfidf, y_valid_tfidf = train_test_split(train_tfidf, train['label'], test_size=0.3, random_state=17)

# Logistic Regression model
def train_and_evaluate(model, x_train, y_train, x_valid, y_valid):
    model.fit(x_train, y_train)
    predictions = model.predict_proba(x_valid)[:, 1] >= 0.3
    predictions = predictions.astype(int)
    return f1_score(y_valid, predictions)

log_reg = LogisticRegression(random_state=0, solver='lbfgs')

# Train and evaluate model on BoW and TF-IDF features
log_bow_score = train_and_evaluate(log_reg, x_train_bow, y_train_bow, x_valid_bow, y_valid_bow)
log_tfidf_score = train_and_evaluate(log_reg, x_train_tfidf, y_train_tfidf, x_valid_tfidf, y_valid_tfidf)

print(f"F1 Score with Bag-of-Words: {log_bow_score}")
print(f"F1 Score with TF-IDF: {log_tfidf_score}")

# Predict labels for the test set
test_tfidf = tfidf[train_size:]
test_predictions = log_reg.predict_proba(test_tfidf)[:, 1] >= 0.3
test['label'] = test_predictions.astype(int)

# Prepare and save the submission file
submission = test[['id', 'label']]
submission.to_csv('result.csv', index=False)
print("Submission file saved as 'result.csv'")
