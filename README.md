# Detecting Fake News with NLP using ML

Predicting whether a News Article is Real or Fake with 99% Accuracy using only the Title and Text, while exploring different NLP Processing techniques and ML Models. 

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
