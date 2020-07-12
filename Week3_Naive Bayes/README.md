The project this week focuses on implementing Naive Bayes for text classification.

My data consists of customers’ reviews on clothing. They are labeled according to their ratings, specifically 1 for rating 4 and 5, and 0 for 3 and below. The goal is to train the model to classify the new data based on reviews.

I introduced only two classes to capture customer’s sentiment partly because the data is skewed towards higher ratings, and a full-range classification might introduce under-fitting. One thing I would like to try in the future is to classify the data differently.

There is additional information in the data that I did not get to use this week, such as the age of the customer, the type of clothes, etc. One potential enhancement in the future would be to introduce extra features into the model.

I implemented a Naive Bayes classifier based on the ‘bag of words’ model. I assumed the occurrence of words followed a multinomial distribution and transformed my text data into a counter vector instead of a binary vector. I tried two different ways to build an NB model. One was to make it from scratch by writing my own functions. The other was to use the Sk-learn package. Sk-Learn package constructed the features in the form of a one-hot vector while my functions adopted a tf-idf vector for feature extraction. 
Noticing the words with the most occurrences are ‘stop words’, I also included a function to exclude some of those words from being selected as features. Sk-Learn makes this even easier via parameter ‘stop_words = English,’ whose library is customizable. I further included a smoothing factor = 1 to take care of words that appeared in the test text but did not make their way into the training set. Both the feature selection and the smoothing factor choice are areas that can be fine-tuned further.

When calculating words occurrence in the training data, I calculated it by dividing the total frequency of the words by the total number of words in the entire sample set. An alternative approach would be to calculate the occurrence per sample take the average. I use the first one as the output is less likely to be skewed by the short reviews. Specifically, the first approach will weigh all reviews equally regardless of their length.

The prediction turns out to be quite fine. Using the sk-learn package, I generated a slightly higher precision and recall than using my functions (0.94,0.94 vs 0.90,0.90). The difference could be related to the choice of ‘stop words,’ how the data gets split, etc, and is worth digging. For both algos, the next step would be to work on the potential tweaks I mentioned earlier to improve the results.
