import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from nltk import pos_tag
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Change if needed
print_analytics = False
real_time = False  # Load pre-processed df from disk if False, Pre-process in Real Time if False

# Importing the datasets
if real_time:
    fake = pd.read_csv('data/Fake.csv')
    true = pd.read_csv('data/True.csv')
else:
    fake = pd.read_pickle('processed_data/fake.pkl')
    true = pd.read_pickle('processed_data/true.pkl')

"""
Exploring the dataset:
We will lemmatize the data and remove stopwords, after which we can find out the most commonly used words in each set.
"""


def pre_processing_1(text):
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()

    text.replace('US', 'united states')
    text = text.lower()  # lowercase
    text = text.replace('u.s.', 'united states')  # US is an important word to keep
    text = re.sub('[^A-Za-z0-9]+', ' ', text)  # Remove links(works on most, but not all: e.g. twitter links don't

    # Remove all symbols, apply lowercase, tokenize
    tokens = word_tokenize(text)

    # Remove stop words and Lemmatize
    text = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    # Turning token list back into string
    text = ' '.join(text)

    return text


def word_analytics(dataframe, column_name, df_name):
    info = dataframe[column_name].str.split(expand=True).stack().value_counts()
    info.name = f'Frequency in {df_name.title()} news'
    info = info.head(100)
    info.to_csv(f'analytics/{df_name}/{df_name}_{column_name}_word_frequency.csv')

    all_text_in_column = ' '.join(dataframe[column_name]).title()
    wc = WordCloud(width=5000, height=4000).generate(all_text_in_column)
    plt.figure(figsize=(50, 40))
    plt.tight_layout(pad=0)
    plt.title(f'WordCloud for {df_name.title()} news')
    plt.imshow(wc)
    plt.axis("off")

    plt.savefig(f'analytics/{df_name}/{column_name}_wordcloud.png', bbox_inches='tight')

    if print_analytics:
        plt.show()
        print(info.head(10))

    plt.close()


if real_time:
    # Processing data
    fake['processed_text'] = fake['text'].apply(lambda text: pre_processing_1(text))
    true['processed_text'] = true['text'].apply(lambda text: pre_processing_1(text))
    fake['processed_title'] = fake['title'].apply(lambda title: pre_processing_1(title))
    true['processed_title'] = true['title'].apply(lambda title: pre_processing_1(title))
    # Save to csv to not pre-process again
    fake.to_pickle('processed_data/fake.pkl')
    true.to_pickle('processed_data/true.pkl')

    word_analytics(fake, 'processed_title', 'fake')
    word_analytics(true, 'processed_title', 'real')
    word_analytics(fake, 'processed_text', 'fake')
    word_analytics(true, 'processed_text', 'real')

    # Merging the True and Fake datasets
    true['label'], fake['label'] = 1, 0
    data = pd.concat([true, fake])
    data.to_pickle('processed_data/processed_dataset.pkl')

"""
POS Filtered Dataset Creation
"""


def POS_filter(list_of_POS, text):  # A function to limit text to certain Parts-of-Speech
    tokens = word_tokenize(text)
    tokens = pos_tag(tokens)
    text = [word for (word, pos) in tokens if pos in list_of_POS]
    # Turning token list back into string
    text = ' '.join(text)
    return text


def generate_pos_filtered_data(source_data, list_of_POS):  # Creating a pkl file of the POS-processed datasets
    new_data = source_data[['processed_text', 'processed_title']].copy()
    new_data['pos_processed_text'] = new_data['processed_text'].apply(
        lambda processed_text: POS_filter(list_of_POS, processed_text))
    new_data['pos_processed_title'] = new_data['processed_title'].apply(
        lambda processed_title: POS_filter(list_of_POS, processed_title))
    new_data.to_pickle(f'processed_data/pos_filtered/{"_".join(map(str, list_of_POS))}_dataset.pkl')


pos_combination_list = [['NN', 'VB'], ['NN', 'JJ'], ['NN'], ['VB', 'JJ']]  # Lists of POS of interest. Not aimed at optimality.

for pos_list in pos_combination_list:
    generate_pos_filtered_data(pd.read_pickle('processed_data/processed_dataset.pkl'), pos_list)
