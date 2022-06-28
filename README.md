# Sentiment-Analysis
## ANALYSIS OF SENTIMENTS OF INDIAN CITIZENS ON NEP 2020 USING TWEETS


### PROBLEM STATEMENT

Despite huge changes in the current scenario and the immense increase in technology and resources available, the Indian education system policy has not been updated since the beginning of 21st century. The government of India has hence come up with NEP 2020 in order to update its policy, to benefit the students and the entire education system as a whole. Changes in education policy can impact the society and the country immensely and utmost care must be taken to consider all the views and effects of the policy. Analyzing people's sentiments and their viewpoints can help make better decisions. So this project aims at analyzing the sentiments of the citizens of India in relation to the national education policy 2020 through the comments of people on twitter ( a most common social media website for public discussions and exchange of opinions).


### DATA COLLECTION

The data for training the model has been collected from Kaggle. The dataset included tweets on NEP 2020 from 31st of July, 2020 to 12th August, 2020. A total of 18240 tweets with respect to 7 features were collected. 

The 7 features included 
-	Unnamed: 0 (a blank column that had no data)
-	Author_ID (ID of the twitter user)
-	Date_of_tweet (the data the tweet was done)
-	Tweet (The text of the tweet)
-	Likes_on_tweet (Number of likes the tweet received)
-	User_handle (The username of the twitter account) 
-	Tweet_link (Link of the tweet)

Source link of the training dataset:

https://www.kaggle.com/datasets/rishabh6377/india-national-education-policy2020-tweets-dataset

For analyzing the sentiments of the citizens, we scraped tweets from Twitter using Twitter API and the ‘tweepy’ package in python. To access the API, a developer account was necessary and we have applied for the elevated access of the account citing the research purpose and security of the data. When the access was granted to the account created using University’s credentials, we established the connection from python to Twitter using tweepy package.
 
The hashtags we used to collect the data were - 
#NationalEducationPolicy2020
#nationaleducationpolicy2020
#nep2020
#NEP2020
#RejectNEP2020
#rejectnep2020
#NewEducationPolicy2020
#neweducationpolicy2020
#recallnep2020
#RecallNEP2020

The scraped data included tweets during the period - 30th March, 2022 to 10th April, 2022. This test dataset consists of 1742 tweets.


### FEATURE MODELING

First we created a feature called ‘Label’ which encodes the Polarity values as 
●	>0 if tweet is positive
●	=0 if tweet is neutral
●	<0if tweet is negative.

Then we create another column ‘sentiment’ which encodes these values as 1, 0 and -1 respectively.

We used VADER Sentiment Analysis to identify the intensity of positive and negative tweets. VADER stands for Valence Aware Dictionary and sEntiment Reasoner which is a lexicon and rule-based sentiment analysis tool. A sentiment lexicon is a list of words which are generally labeled according to their semantic orientation as either positive or negative. VADER not only tells about the Positivity and Negativity score but also tells us about how positive or negative a sentiment is.

We created 3 features called - Positive, Negative and Neutral to calculate the intensity of sentiment of a particular tweet. It was observed that the sentiment score for Neutral tweets was the highest.

Next, we have considered X as the lemmatized tweets of data and y as sentiment value. Train_test_split function in sklearn was used to split the data into training and validation dataset in the ratio of 80% and 20%.

TF-IDF (Term Frequency Inverse Document Frequency) was used to vectorize the tweets (X_train and X_val). It is available in “sklearn.feature_extraction.text” which is used to transform text into a meaningful representation of numbers which is then used to fit machine algorithms for prediction or classification.
  
It is to be understood that Countvectorizer gives the number of frequencies with respect to the index of vocabulary whereas tfidf considers overall documents of weight of words.  



The formula is a combination of both TF(Term Frequency) and IDF(Inverse Document Frequency). TF is the number of times a term appears in a particular document which is specific to a document. It can be calculated as,


tf(t) = No. of times term ‘t’ occurs in a document


IDF is a measure of how common or rare a term is across the entire corpus of documents. The point to be noted here is that this is common to all the documents so if a word is common and appears in many documents, then the idf value will approach 0 else it approaches 1 if it’s rare. It can be calculated as,

idf(t) = log e [ (1+n) / ( 1 + df(t) ) ] + 1 (default i:e smooth_idf = True)
and
idf(t) = log e [ n / df(t) ] + 1 (when smooth_idf = False)
The tfidf value of a term in a document is the product of its tf and idf and higher the value, the more relevant the term is in that document.

Viewing the target variable ‘sentiment’, we observed that many of the tweets correspond to positive class. As a result, we may deduce that there is a class imbalance problem, which must be addressed prior to the model training stage in order for the model to be free of bias toward the most common class. To solve this issue, we used SMOTE(Synthetic Minority Over-Sampling Technique).

SMOTE is an over-sampling method that over-samples the minority class by producing "synthetic" instances rather than over-sampling with replacement. This implies that when SMOTE generates new synthetic data, it will choose one data to duplicate and examine its k closest neighbors. It will then generate random values in feature space that are between the original sample and its neighbors. One important thing is that the oversampler won’t be able to handle raw text data. So we first transformed it into a feature space using TfidfVectorizer and then applied SMOTE on it.


### MODEL BUILDING AND EVALUATION

Various classification models like Multinomial Naive Bayes, Multinomial Logistic Regression, Support Vector Machine and Random Forest Classifier have been used to classify the tweets as positive, neutral or negative.

As the target variable has three classes - positive, neutral and negative, traditional Naive Bayes and Logistic Regression cannot be used as they deal with binary classification. 

●	Multinomial Naive Bayes 

The Multinomial Naive Bayes method is a Bayesian learning model that is commonly used in Natural Language Processing (NLP) especially with problems regarding multiple classes. The program guesses the tag of a text, such as an email or a newspaper story by calculating  each tag's likelihood for a given sample and outputs the tag with the greatest chance.In text analytics, where samples are generated from a common lexicon, this strategy is particularly effective.

According to scikit learn, Multinomial Naive Bayes Classifier is suitable for classification with discrete features (e.g., word counts for text classification). The multinomial distribution normally requires integer feature counts but however, in practice, fractional counts such as tf-idf may also work.

Multinomial Naive Bayes takes into account a feature vector in which each term denotes the number of times it appears or how frequently it appears, i.e. frequency. Here, we assume the ‘naive’ condition that every word in a sentence is independent of the other ones.

 It's based on the Bayes formula :   P(A|B) = P(A) * P(B|A)/P(B)

where A is the class of the possible outcomes and B is the given instance which has to be classified, representing some certain features.

We have implemented this technique calling upon the inbuilt function from scikit-learn. 

The accuracy obtained after fitting the model on training data is 71.28%

By using 10-fold Cross Validation, the accuracy obtained is 85.19% with an F1 score of 85.03%. 

●	Multinomial Logistic Regression

Multinomial logistic regression is an extension of logistic regression that provides support for multiclass classification problems. By default, logistic regression is confined to two-class classification tasks. Some extensions, such as one-vs-rest, can be used to solve multi-class classification problems with logistic regression, but they need the classification problem to be split into many binary classification problems first. The multinomial logistic regression algorithm, on the other hand, which is an extension of the logistic regression model involves changing the loss function to cross-entropy loss and predicting the probability distribution to a multinomial probability distribution to support multi-class classification problems. 

In simple words, it can be said that Multinomial Logistic Regression is a modified version of logistic regression that predicts a multinomial probability (i.e. more than two classes) for each input example. Sigmoid of Logit function is used in logistic regression model for binary classification whereas Softmax function is used for multi classification. The Softmax function is a probabilistic function which calculates the probabilities for the given score and returns the high probability value for the high scores and fewer probabilities for the remaining scores.

The accuracy obtained after fitting Multinomial Logistic Regression on training data is 81%

By using 10-fold Cross Validation, the accuracy obtained is 91.41% with an F1 score of 91.31%. 

●	Random Forest Classifier

A random forest is an ensemble classifier that makes predictions using a variety of decision trees. It works by fitting a number of decision tree classifiers to different sub - samples of the dataset. It builds decision trees on several samples and takes their majority vote for classification. The greater number of trees in the forest leads to higher accuracy and prevents the problem of overfitting.One of the most important features of the Random Forest Algorithm is that it can handle the data set containing continuous variables as in the case of regression and categorical variables as in the case of classification. 

The accuracy obtained upon fitting Random Forest Classifier on data is 74%

By using 10-fold Cross Validation, the accuracy obtained is 90.11% with an F1 score of 85.52%. 

●	Support Vector Machine

SVM stands for Support Vector Machine and is a supervised machine algorithm used for classification and regression. SVM classifies data by finding a hyperplane that establishes a boundary between two classes of data. As the name implies, a hyperplane is a plane in n-dimensional space that divides data into classes or groups.SVM tries to fit an optimal hyperplane rather than simply any hyperplane.

The aim of SVM is to identify a hyperplane in an n-dimensional space that maximises the separation of data points from their actual classes. The data points which are at the minimum distance to the hyperplane i.e, closest points are called Support Vectors. SVM does not enable multiclass classification in its most basic form. After breaking down the multi-classification problem into smaller subproblems, all of which are binary classification problems, the same method is used for multiclass classification.

The accuracy obtained upon fitting Random Forest Classifier on data is 83%

By using 10-fold Cross Validation, the accuracy obtained is 95.65% with an F1 score of 93.43%.


### CONCLUSION

From our sentiment analysis on National Education Policy 2020 (NEP2020) using twitter data, we conclude that the majority of Indian citizens have either a positive or a neutral opinion towards the policy. Most of the people believe that it is a best or excellent policy being considered by the Government of India and are in support of the policy implementation. Although the overall response towards the policy can be considered as favorable, there are a few people who are strongly opposing it. The intensity of their opposition towards the policy is high and cannot be neglected; hence not just the majority supporting class but the viewpoints of the minority opposing class should also be considered for this education policy to be a success and bring a positive impact on the Indian education system and the nation as a whole.
