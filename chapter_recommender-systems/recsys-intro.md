# Overview of Recommender Systems



In the last decade, the Internet has evolved into a platform for large-scale online services, which profoundly changed the way we communicate, read news, buy products, watch movies, etc.  In the meanwhile, the unprecedented number of items (we use the term item to refer movies, news, books, products, etc.) offered online requires a systems that can help us discover items we preferred. Recommender systems are such information filtering tools that can facilitate personalized services and provide tailored experience.  They play a pivotal role in utilizing the wealth of data available to make choices manageable. Nowadays, recommender systems are at the core of  a number of online services providers such as Amazon, Netflix, YouTube, etc. (recall the example of Deep learning books recommended by Amazon in Section [1.3.1.5](http://numpy.d2l.ai/chapter_introduction/intro.html#recommender-systems)  The benefits of employing recommender systems are two-folds: Firstly, it can largely reduce user's effort in finding items and alleviate the issue of information overload. Secondly, it can add business value to  online
service providers and is a important of source of revenue.  This chapter will introduces the fundamental concepts, classic models and recent advances with deep learning in the field of recommender systems, together with examples implemented by  MXNet.



## Collaborative Filtering

We start the journey with the important concept in recommender systems - collaborative filtering
(CF), which was firstly coined by the [Tapestry](https://dl.acm.org/citation.cfm?id=138867) system, referring to "people collaborate to help one another perform the filtering process  in order to handle the large amounts of email and messages posted to newsgroups". This term has been enriched with more senses. In a broad sense, it is the process of
filtering for information or patterns using techniques involving collaboration among multiple users, agents, data sources, etc.   CF has many forms and numerous CF methods proposed since its advent.  

Overall, CF techniques can be categorized into: memory-based CF, model-based CF and their hybrid.
Representative memory-based CF techniques are nearest neighbor-based CF such as user-based CF and item-based CF.  Latent factor models such as matrix factorization are examples of model-based CF.  Memory-based CF has limitations in dealing with sparse and large-scale data since it computes the similarity values based on common items.  Model-based methods become more popular with its
better capability in dealing with sparsity and scalability.  Many model-based CF approaches can be extended with neural networks, leading to more flexible and scalable models with the computation acceleration in deep learning.  In general, CF only uses the user-item interactions data to make predictions and recommendations. Besides CF, content-based and context-based recommender systems are also useful in incorporating the content descriptions of items/users and contextual signals such as timestamp and locations.  Obviously, we may need to adjust the model types/structures with different input data available. 



## Explicit Feedback and Implicit Feedback 

To know user's preference, the system shall collect feedback from them.  The feedback can be either explicit or implicit. For example,  [IMDB](https://www.imdb.com/) collects star ratings range from one to ten stars for movies. YouTube provides the thumbs-up and thumbs-down buttons for users to show their preferences.  It is apparent that
gathering explicit feedback requires users to indicate their interests proactively.  Nonetheless,  explicit feedback is not always available as many users may be reluctant to rate products or gathering it is too expensive.  As such, many recommender systems are centered on implicit feedback which
indirectly reflects user's opinion through observing user behavior.   There are diverse forms of implicit feedback inculde purchase history, browsing history, watches and even mouse movements. For example, a user that purchased many books by the same author probably likes that author.   Note that implicit feedback is inherently noisy.  We can only *guess* their preferences and true motives. A user watched a movie does not necessarily indicate a positive view of that movie.



## Recommendation Tasks 

A number of recommendation tasks have been investigated in the past decade.  Based on the domain of applications, there are movies recommendation, news recommendations, point-of-interest recommendation and so forth.  It is also possible to differentiate the tasks based on the types of feedback and input data, for example, the rating prediction task aims to predict the explicit ratings. Top-n recommendation (collaborative ranking) ranks all items for each users personally based on the implicit feedback. If timestamp information is also included, we can build sequence-aware recommendation.  Another popular task is called click-through rate prediction, which is also based on implicit feedback, but various categorical features can be utilized.  



## Summary

* Recommender systems are important for individual users and industries. Collaborative filtering is a key concept in recommendation.
* There are two types of feedbacks: implicit feedback and explicit feedback.  A number recommendation tasks have been explored during the last decade.

## Exercises

1. Can you explain how recommender systems influence your daily life ?
2. What interesting recommendation tasks do you think can be investigated ?
