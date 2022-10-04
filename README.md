# Detecting Fake News with NLP using ML

Predicting whether a News Article is Real or Fake with 99.6% Accuracy using only the Title and Text, while exploring different NLP Processing techniques and ML Models. 

>Dataset: [Fake + Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

## preprocessing.py
__Real Time Data Loading and Processing:__

Use __print_analytics__ to view information on the input dataset and WordClouds in the terminal. 
To load and process data in real time, enable __real_time__. 
```python
print_analytics = False
real_time = False 
```

__POS-Tagging__:

To create new datasets where text is filtered with desired Parts-Of-Speech, modify the list below.
```python
pos_combination_list = [['NN', 'VB'], ['NN', 'JJ'], ['NN'], ['VB', 'JJ']]
```
[List of Parts-of-Speech and their abbreviations](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)
## train.py
__Training Models with different NLP Techniques:__

Use __generate_reports__ function to create a CSV with Accuracy, Recall and Precision for each combination of the following inputs:

>__Vectorizer:__ the feature of choice for converting text data to numerical input.

>__Model:__ Machine Learning models.

>__Data:__ can be filtered to contain specific POS. Generated after __preprocessing.py__ has been executed.

To accomodate multiple combinations, use dictionaries for input to the __generate_reports__ function, as shown below.

```python
vectorizers = {
    'Name of Vectorizer': Vectorizer Object,  # Format
    'TFIDF': TfidfVectorizer(min_df=10) } # Example

models = {
    'Model Name': Model Object, 
    'Logistic_Regression': LogisticRegression() }

filtered_data = {
    'Filter Name': 'Path to filtered Data', # Created by preprocessing.py
    'Noun_Adjective': 'processed_data/pos_filtered/NN_JJ_dataset.pkl' }

generate_reports(vectorizers, models, filtered_data)
```

## Directories
>__Data:__ [Original Fake + Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

>__Analytics:__ WordClouds with Word Frequency and CSVs containing top 100 most used words for each dataset.

>__Model_Results:__ Confusion Matrices and a report on Accuracy, Precision and Recall for each combination of Models, Filters and Vectorizers - __result.csv__.

## Results
__No POS Filter:__

|ML Model           |Feature        |Precision     |Recall     |Accuracy   |
|-------------------|---------------|--------------|-----------|-----------|
|Logistic Regression|Frequency Count|0.996         |0.997      |0.996      |
|Random Forest      |Frequency Count|0.995         |0.993      |0.994      |
|Random Forest      |TFIDF          |0.991         |0.990      |0.991      |
|Logistic Regression|TFIDF          |0.983         |0.988      |0.986      |
|Naive Bayes        |Frequency Count|0.945         |0.952      |0.951      |
|Naive Bayes        |TFIDF          |0.931         |0.936      |0.936      |


__Top 5 Best Filtered Results:__

|ML Model           |Feature        |Filter        |Precision  |Recall     |Accuracy   |
|-------------------|---------------|--------------|-----------|-----------|-----------|
|Logistic Regression|Frequency_Count|Noun Adjective|0.981      |0.974      |0.978      |
|Logistic Regression|TFIDF          |Noun Adjective|0.970      |0.971      |0.972      |
|Logistic Regression|Frequency Count|Noun Verb     |0.975      |0.964      |0.971      |
|Random Forest      |Frequency Count|Noun Adjective|0.973      |0.964      |0.970      |
|Logistic Regression|Frequency Count|Noun          |0.970      |0.967      |0.970      |

__Top 5 Worst Overall Results:__

|ML Model   |Feature        |Filter        |Precision|Recall|Accuracy|
|-----------|---------------|--------------|---------|------|--------|
|Naive Bayes|TFIDF          |Noun Adjective|0.923    |0.907 |0.920   |
|Naive Bayes|Frequency Count|Noun          |0.918    |0.908 |0.917   |
|Naive Bayes|TFIDF          |Verb Adjective|0.914    |0.906 |0.914   |
|Naive Bayes|TFIDF          |Noun Verb     |0.917    |0.895 |0.911   |
|Naive Bayes|TFIDF          |Noun          |0.911    |0.896 |0.908   |

