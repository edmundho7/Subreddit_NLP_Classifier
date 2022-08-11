# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 3: Reddit NLP Classification

### Overview

In this project, we are tasked to create a subreddit classification model by utilising Natural Language Processing (NLP), Data wrangling, APIs and our knowledge of the different classification models.
As usual, the data science workflow will be carried out in order to answer this classification problem.
1. Problem Statement
2. Data Collection
3. Exploratory Data Analysis
4. Modelling and model selection
5. Recommendation and conclusion

### Datasets
The datasets are collected from Reddit using [Pushshift's](https://github.com/pushshift/api) API to collect posts from the two subreddits.

There are 3 datasets in the [`data`](./data/) folder for this project.

* [`wallstreetbets.csv`](./data/wallstreetbets.csv): 2000 posts scrapped from r/wallstreetbets
* [`stocks.csv`](./data/stocks.csv): 2000 posts scrapped from r/stocks
* [`combined_cleaned.csv`](./combined_clean.csv): Combined dataset of both wallstreetbets and stocks after data cleaning 

### Data dictionary

r/wallstreetbets and r/stocks
| Feature   	| Type   | Description                                                                                                                                
|---------------|--------|----------------------------------------------------------------------------------------------|
| subreddit 	| object | The subreddit where the post was taken from. This is either 'wallstreetbets' or 'stocks'.    | 
| selftext 	| object | Body text of the subreddit post  								|					
| title     	| object | Title of the post 										|					
| created_date  | object | The datetime of the created post 								|					

Combined cleaned data
| Feature   	      | Type   | Description                                                                                  |                                         
|---------------------|--------|----------------------------------------------------------------------------------------------|
| subreddit 	      | object | The subreddit where the post was taken from. This is either 'wallstreetbets' or 'stocks'.    |
| selftext 	      | object | Body text of the subreddit post  							      |			
| title     	      | object | Title of the post 									      |					
| combined_text       | object | Combined text of selftext and title. Duplicates and text cleaning has been performed         |
| combined_text_length| object | Length of combined text								      |
| combined_text_count | object | Word count of combined text								      |
| title_count	      | object | Word count of Title 								              |
| token_combined_text | object | Tokenized text of combined text							      |
| lem_combined_text   | object | Lemmatized tokenized text								      |
| stem_combined_text  | object | Porter stemmed tokenized text								      |

---

### Problem statement and Executive Summary

Financial advice and discussions are rife on the internet nowadays and available to almost everyone. Reddit is a social media platform where users can come together to discuss on their topic of interest on the different subreddits. 
The subreddits that I am going to use for this NLP classification project is the infamous r/wallstreetbets and r/stocks where users discuss on trading strategies and on the stocks that they are interested in. 

The goal of the project is two-fold:
1. Using [Pushshift's](https://github.com/pushshift/api) API to collect posts from two subreddits
2. Create a NLP classifier to allow users to distinguish posts from these two subreddits in order to make informed trading decisions.

**Methodology**

The project was completed in 3 separate notebooks. 
1. Data Collection
	- Webscraping of data from Reddit using Pushshift API
	- Convert epoch time to datetime
	- Drop irrelevant columns

2. Data cleaning & EDA
	- Combine the 2 dataset together
	- Perform data cleaning (Impute null data, combine columns, create additional columns, drop duplicates)
	- Feature engineering
	- Create stop worrd list
	- Tokenizing, stemming and lemmatizating of text data
	- Create distribution chart of word count and length
	- Explore Count Vectorizer and TF-IDF Vectorizer to identifiy common words and n-grams of datasets

3. Modelling 
	- Perform train, test split on final dataset of unmodified, stemmed and lemmatized text
	- Create pipeline models and parameters
	- Run models through these pipelines and parameters
	- Evaluate performance metric and select the best model for the dataset
	
**Model comparison**

Hyperparameter tuning was performed on the different models to get the best parameters for each models. These models are then evaluated based on several metrics and criteria. The results are shown in the table below.

|                                           | Train score | Test score | Generalisation | Accuracy | Precision | Recall | Specificity |    F1 | ROC AUC |
|------------------------------------------:|------------:|-----------:|---------------:|---------:|----------:|-------:|------------:|------:|--------:|
|  Random Forest Count Vectorizer Unmodifed |       0.725 |      0.699 |          3.586 |    0.699 |     0.676 |  0.812 |       0.577 | 0.738 |   0.694 |
| Random Forest Count Vectorizer Lemmatized |       0.711 |      0.688 |          3.235 |    0.688 |     0.684 |  0.746 |       0.626 | 0.714 |   0.686 |
|    Random Forest Count Vectorizer Stemmed |       0.724 |      0.685 |          5.387 |    0.685 |     0.670 |  0.779 |       0.582 | 0.720 |   0.680 |
|            Random Forest TF-IDF Unmodifed |       0.744 |      0.699 |          6.048 |    0.699 |     0.677 |  0.807 |       0.582 | 0.736 |   0.694 |
|           Random Forest TF-IDF Lemmatized |       0.727 |      0.691 |          4.952 |    0.691 |     0.692 |  0.734 |       0.645 | 0.712 |   0.689 |
|              Random Forest TF-IDF Stemmed |       0.762 |      0.694 |          8.924 |    0.694 |     0.683 |  0.769 |       0.612 | 0.723 |   0.690 |
|     Naive Bayes CountVectorizer Unmodifed |       0.702 |      0.709 |         -0.997 |    0.709 |     0.730 |  0.701 |       0.719 | 0.715 |   0.710 |
|    Naive Bayes CountVectorizer Lemmatized |       0.693 |      0.692 |          0.144 |    0.692 |     0.705 |  0.704 |       0.680 | 0.704 |   0.692 |
|       Naive Bayes CountVectorizer Stemmed |       0.702 |      0.703 |         -0.142 |    0.703 |     0.710 |  0.726 |       0.678 | 0.718 |   0.702 |
|              Naive Bayes TF-IDF Unmodifed |       0.751 |      0.728 |          3.063 |    0.728 |     0.815 |  0.618 |       0.847 | 0.703 |   0.733 |
|             Naive Bayes TF-IDF Lemmatized |       0.745 |      0.713 |          4.295 |    0.713 |     0.788 |  0.616 |       0.820 | 0.691 |   0.718 |
|                Naive Bayes TF-IDF Stemmed |       0.758 |      0.712 |          6.069 |    0.712 |     0.771 |  0.636 |       0.795 | 0.697 |   0.715 |

By evaluating based on accuracy and precision as the main metric, the Naive Bayes Model with TF-IDF preprocessing on unmodified text is selected to be the most suitable model for the dataset

### Recommendation & Conclusion

The Naïve Bayes Model with TF-IDF Vectorizer on Unmodified text will be the final model selected for the subreddit classifier as it is the best performing model with its accuracy, precision and AUC score.

The following can be explored to further improve the model accuracy and to find the best parameters.
- More hyperparameter tuning and classifiers can be explored to further increase the accuracy of the classifier
- More weightage can be assigned to Title than self text
- As there are high frequency postings in both r/wallstreetbets and r/stocks, we can consider to scrape only popular posts in order to improve the quality of the scrapped data and reduce spam
- Explore sentimental analysis to the model to determine the positive and negative sentiments of the posts
- Develop a list of popular tickers and stop words to add to the stopwords corpus


