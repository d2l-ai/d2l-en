# Introduction to Deep Learning

You may have some experience with programming and have developed one or two programs. At the same time, you may have read overwhelming reports about deep learning or machine learning, although many times they are given a broader name: artificial intelligence. In fact, or fortunately, most programs do not require deep learning or artificial intelligence techniques in a broader sense. For example, if we have to write a user interface for a microwave oven, with a little effort we can design a dozen buttons and a series of rules that accurately describe the performance of the microwave oven under various conditions. Or if we're going to program an email client, this program will be more complicated than a microwave oven, but we can still concentrate and think step by step: the client user interface will need several input boxes to accept recipients, subjects, and mail body, etc., and the program will listen to the keyboard inputs and write to a buffer, and then display them in the corresponding input box. When the user clicks the "Send" button, we need to check if the format of the recipient's email address is legitimate, and check if the subject of the email is empty, or warn the user when the subject is empty, and then use the corresponding protocol to deliver the email.

It is worth noting that in the above two examples, we do not need to collect data in the real world, nor do we need to systematically extract the feature of these data. As long as there is plenty of time, our common sense and programming skills are enough for us to complete them.

At the same time, it's easy to find some simple questions that even the best programmers in the world can't solve with only programming skills. For example, suppose we want to write a program that determines if there is a cat in an image. It sounds simple, right? The program only needs to output "true" (indicating a cat) or "false" (indicating no cat) for each input image. Surprisingly, even the best computer scientists and programmers in the world don't know how to write such a program.

Where should we start? First, let's further simplify this problem: if all the images are assumed to have the same size of the height and the width of 400 pixels, and one pixel consists of three values of red, green and blue, then an image is represented by nearly 500,000 numerical values. So which values hide the information we need? Is it the average of all the values, the values of the four corners, or is it a particular point in the image? In fact, to interpret the content in an image, you need to look for features that only appear when you combine thousands of values, such as edges, textures, shapes, eyes, noses, etc., in order to determine whether the image contains a cat.

One way to solve the above problem is to think backwards. Rather than designing a problem-solving program, we might as well start looking for a solution from the final need. In fact, this is also the core idea shared between the current machine learning and deep learning applications: we can call it "programming with data." Instead of sitting in the room and thinking about how to design a program to recognize a cat, it is better to utilize the ability of human eyes to recognize a cat in the image. We can collect real images that are known to contain a cat and no cat, and then our goal translates into how to get started with these images to get a function that can infer whether the image contains a cat. The form of this function is usually chosen by our knowledge for a specific problem: for example, we use a quadratic function to determine if the image contains a cat. However, the specific value of a function parameter such as a quadratic function coefficient value is determined by data.

In general, machine learning is a discipline that discusses a wide variety of functional forms that apply to different problems and how to use data to efficiently obtain specific values of function parameters. Deep learning refers to a class of functions in machine learning, usually in the form of multilayer neural networks. In recent years, relying on big data sets and powerful hardware, deep learning has gradually become the main method for processing complex high-dimensional data such as images, text corpora and sound signals.

We are now in an era of programming with more and more help from deep learning. This can be said to be a turning point in the history of computer science. For example, deep learning is already used in your phone: spell correction, speech recognition, recognition of friends in social media photos, and more. Thanks to excellent algorithms, fast and inexpensive computing power, unprecedented amounts of data, and powerful software tools, most software engineers today have the ability to build complex models to solve problems that even the best scientists found tricky ten years ago.

This book hopes to help you enter the wave of deep learning. We hope that deep learning will be at your fingertips through the combination of math, codes and examples. This book does not require that you  have a deep background in mathematics and programming, and we will explain the necessary knowledge step by step as the chapters roll out. It is worth mentioning that each section of the book is a Jupyter notebook that can run independently. You can get these notebooks online and execute them on your PC or cloud server. This way you can change the codes in the book and get timely feedback. We hope this book will help and inspire a new generation of programmers, entrepreneurs, statisticians, biologists, and anyone interested in using deep learning to solve problems.


## Origin

Although deep learning seems to be the term that has just emerged in recent years, the core idea of using data programming and neural networks, upon which deep learning is based, has been studied for hundreds of years. Since ancient times, humans have been eager to analyze the data in order to know how to predict the future. In fact, data analysis is the essence of most natural sciences. We want to extract rules from everyday observations and look for uncertainty.

As early as the 17th century, [Jacob Bernoulli (1655-1705)](https://en.wikipedia.org/wiki/Jacob_Bernoulli) proposed a Bernoulli distribution, describing a stochastic process with only two outcomes (such as throwing a coin). About a century later, [Carl Friedrich Gauss (1777-1855)](https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss) invented the least squares method that is still widely used today in the fields from insurance calculation to medical diagnosis. Tools such as probability theory, statistics, and pattern recognition helped natural science experimenters regress from data to natural laws, discovering a series of natural laws that are perfectly expressed by linear models, such as Ohm's law (the law describing the relationship between the voltage across a resistor and the current flowing through the resistor).

Even in the Middle Ages, mathematicians were keen to use statistics to make estimates. For example, [Jacob Kobels' (1460-1533)](https://www.maa.org/press/periodicals/convergence/mathematical-treasures-jacob-kobels-geometry) geometric book documented using the average foot length of 16 men to estimate the average foot length of a man.

![In the Middle Ages, the average foot length of 16 men was used to estimate the average foot length of a man. ](../img/koebel.jpg)

As shown in Figure 1.1, in this study, 16 adult men were asked to stand in a row and put their feet together when leaving the church, and then the total length of their feet was divided by 16 to get an estimate: this number is about one foot today. This algorithm was later modified to account for the specific shape of a foot: the longest and shortest feet were not counted, only took the average of the remaining feet length, that is, the prototype of the trimmed mean.

The real take-off of modern statistics in the 20th century was attributable to the collection and distribution of data. [Ronald Fisher (1890-1962)](https://en.wikipedia.org/wiki/Ronald_Fisher), one of the masters of statistics, contributed hugely to the statistical theory and the application of statistics in genetics. Many of the algorithms and formulas he invented, such as linear discriminant analysis and Fisher information, are still frequently used. Even the Iris data set he published in 1936 is occasionally used to demonstrate machine learning algorithms.

The information theory of [Claude Shannon (1916-2001)](https://en.wikipedia.org/wiki/Claude_Shannon) and the computational theory of [Allan Turing (1912-1954)](https://en.wikipedia.org/wiki/Allan_Turing) also have a profound impact on machine learning. Turing put forward in his famous paper [Computing Machinery and Intelligence](https://www.jstor.org/stable/2251299) a question,  "can machines think?"[1]. In the "Turing Test" he described, if a person can't distinguish whether his conversation object is human or machine when using textual interaction, then the machine can be considered intelligent. Today, the development of intelligent machine is changing with each passing day.

Another area that has a major impact on deep learning is neuroscience and psychology. Since human beings are clearly able to demonstrate intelligence, it is reasonable for us to explore the explanation and the reverse engineering of the mechanism of the human intelligence. One of the earliest algorithms was formally proposed by [Donald Hebb (1904-1985)](https://en.wikipedia.org/wiki/Donald_O._Hebb). In his groundbreaking book, [The Organization of Behavior](http://s-f-walker.org.uk/pubsebooks/pdfs/The_Organization_of_Behavior-Donald_O._Hebb.pdf), he suggested that nerves learn through positive reinforcement, i.e. the Hebb Theory[2]. The Hebb Theory is the prototype of the perceptron learning algorithm and has become the cornerstone of many stochastic gradient descent algorithms that support today's deep learning: strengthening desired behavior, punishing undesired behavior, and finally, obtaining excellent neural network parameters.

Biology inspired the name of neural network. Such researchers can be traced back to [Alexander Bain (1818-1903)](https://en.wikipedia.org/wiki/Alexander_Bain) and [Charles Scott Sherrington (1857-1952)](https://en.wikipedia.org/wiki/Charles_Scott_Sherrington) more than a century ago. Researchers have attempted to build computational circuits that mimic neuronal interactions. Over time, the biological interpretation of the neural network was diluted, but the name has been retained. Today, most neural networks contain the following core principles:

* Linear and nonlinear processing units, often referred to as "layers", are used alternately. 
* Use the chain rule (i.e. back-propagation) to update the parameters of the network.

After the initial rapid development, from about 1995 to 2005, most of the machine learning researchers' attention moved away from the neural network. This is due to a variety of reasons. First, training neural networks requires a lot of computing power. Although the memory was sufficient at the end of the last century, the computing power was not enough. Second, the data set used at the time was also relatively small. Fisher's Iris data set, released in 1932, had only 150 examples and was widely used to test the performance of algorithms. The MNIST data set with 60,000 examples was considered very large at the time, although it is now considered to be a typical simple data set. Due to the scarcity of data and computing power, empirically, tools such as the kernel method, the decision tree and the probabilistic graphical model were better. They do not require long-term training like neural networks and provide predictable results under strong theoretical guarantees.

## Development

The rise of the Internet, cost-effective sensors and low-cost memory make it easier for us to access large amounts of data. Coupled with the cheap computing power, especially the GPUs originally designed for computer games, the situation described above has changed a lot. In an instant, algorithms and models that were originally thought impossible are at your fingertips. This trend is evidenced by the following table:

|Year|Number of Data Examples|Memory|Floating Point Calculations per Second|
|:--|:-:|:-:|:-:|
|1970|100 (Iris)|1 KB|100 K (Intel 8080)|
|1980|1 K (Boston house prices)|100 KB|1 M (Intel 80186)|
|1990|10 K (handwritten character recognition)|10 MB|10 M (Intel 80486)|
|2000|10 M (web page)|100 MB|1 G (Intel Core)|
|2010|10 G (advertising)|1 GB|1 T (Nvidia C2050)|
|2020|1 T (social network)|100 GB|1 P (Nvidia DGX-2)|

Obviously, storage capacity has not kept pace with the growth in data volume. At the same time, the increase in computing power has overshadowed the increase in data volume. Such a trend has allowed statistical models to invest more computational power in optimizing parameters, but at the same time it is necessary to improve storage utilization efficiency, such as using nonlinear elements. This has led to the corresponding preferred choice of machine learning and statistics from generalized linear models and kernel methods to deep multilayer neural networks. Such changes are the reasons that the pillar models of deep learning, such as multilayer perceptrons, convolutional neural networks, long- and short-term memory-cycle neural networks, and Q-learning, have been "rediscovered" in the past decade after decades of being disregarded.

Recent advances in statistical models, applications, and algorithms have often been compared to the Cambrian explosion (a period in which the number of species in history exploded). But these advances are not just because we have more resources available to allow us to present the old content in a new form. The following list only covers some of the reasons for the significant development of deep learning in the past decade:

* Excellent capacity control methods, such as dropout, make the training of large networks no longer subject to overfitting (the behavior of large neural network learning to memorize most training data) [3]. This is done by injecting noise into the entire network, such as randomly changing the weight to a random number during training [4].

* The attention mechanism solves another problem that has plagued statistics for more than a century: how to expand the memory capacity and complexity of a system without adding parameters. The attention mechanism uses a learnable pointer structure to construct a sophisticated solution [5]. That is to say, instead of memorizing the entire sentence in a task like machine translation, it is better to memorize the pointer to the intermediate state of the translation. Since the information of the entire original text does not need to be stored before the translation is generated, such a structure makes it possible to accurately translate long sentences.

* Multilevel designs such as memory networks [6] and neural encoder-interpreters [7] make the iterative modeling methods possible for inference processes. These models allow you to repeatedly modify the internal state of the deep network so that you can simulate the steps on the inference chain as if the processor were modifying memory during the computation.

* Another major development is the invention of the generative adversarial network [8]. Traditionally, statistical methods used in probability distribution estimation and generative models have focused more on finding the correct probability distribution and the correct sampling algorithm. A key innovation in the generative adversarial network is to replace the sampled part with any algorithm that contains differentiable parameters. These parameters will be trained so that the discriminator can no longer distinguish between real and generated examples. The feature of generative adversarial network using any algorithm to generate an output opens up a new door for many tricks. For example, the generation of running zebras[9] and the generation of celebrity photos[10] are both testimony to the development of the generative adversarial network.

* In many cases, a single GPU is no longer sufficient for training on large data sets. Our ability to build distributed parallel training algorithms has improved dramatically over the past decade. The biggest bottleneck in designing scalable algorithms is at the heart of the deep learning optimization algorithm: stochastic gradient descent requires relatively smaller batches. At the same time, smaller batches will also reduce the efficiency of the GPU. If we use 1024 GPUs, each GPU has a batch size of 32 examples, then the batch size of the single-step training will be more than 32,000. In recent years, the work of Li Mu [11], Yang You, et al [12], and Xianyan Jia, et al [13], has pushed the batch size to as many as 64,000 examples and reduced the time to train the ResNet-50 model on the ImageNet data set to 7 minutes. In contrast, the initial training time needed to be calculated in days.

* The ability of parallel computing also has contributed to the development of reinforcement learning where the simulation, at least, can be utilized. Parallel computing helps computers surpass levels of human maneuvers in Go, Atari games, StarCraft, and physical simulation.

* The deep learning framework also plays an important role in the process of disseminating deep learning ideas. For example, the first generation frameworks such as [Caffe](https://github.com/BVLC/caffe), [Torch](https://github.com/torch) and [Theano](https://github.com/Theano/Theano), made modeling a simple thing. These frameworks have been used in many groundbreaking papers. They are now have been replaced by [TensorFlow](https://github.com/tensorflow/tensorflow) (often used in the form of high level API [Keras](https://github.com/keras-team/keras)), [CNTK](https://github.com/Microsoft/CNTK), [Caffe 2](https://github.com/caffe2/caffe2) and [Apache MXNet](https://github.com/apache/incubator-mxnet). The third generation, i.e. the imperative deep learning framework, was pioneered by [Chainer](https://github.com/chainer/chainer) that defined the model with a syntax similar to NumPy. This idea was later adopted by [PyTorch](https://github.com/pytorch/pytorch) and MXNet's [Gluon API](https://github.com/apache/incubator-mxnet), the latter being the tool used to teach deep learning in this book.

System researchers are responsible for building better tools and statisticians building better models. This division of labor greatly simplifies the work. For example, in 2014, training a logistic regression model was an assignment given by Carnegie Mellon University to the new entry doctoral students studying machine learning. Today, this problem can be completed with less than 10 lines of code, which can be done by ordinary programmers.

## Successful Cases

For a long time, machine learning has achieved the goal that other methods are difficult to achieve. For example, since the 1990s, the sorting of mail has begun to use optical character recognition. In fact, this is the source of the well-known MNIST and USPS handwritten digital data sets. Machine learning is also the backbone of an electronic payment system that can be used to read bank checks, perform credit ratings, and prevent financial fraud. Machine learning algorithms are used on the web to provide search results, personalized recommendations, and page sorting. Despite not being in the public eye for a long time, machine learning has penetrated into every aspect of our work and life. Until recently, machine learning has gradually become the focus of the public due to the breakthroughs on the problems that were once considered unsolvable and that are directly related to consumers. These advances are largely due to deep learning:

* Intelligent assistants like Apple's Siri, Amazon's Alexa, and Google Assistant can answer verbal questions with considerable accuracy, ranging from simple light turning on/off (which helps the disabled individuals very much) to assisting voice conversations. The emergence of intelligent assistants may serve as a sign that artificial intelligence is beginning to influence our lives.

* The key to intelligent assistants is the need to be able to accurately recognize speech, and the accuracy of some applications of such systems has gradually grown to be neck and neck with humans[14].

* Object recognition has also undergone a long process of development. In 2010, identifying the category of objects from images was still a rather challenging task. In the same year, the top five error rate of 28% was achieved in the ImageNet baseline test [15]. By 2017, this number has dropped to 2.25% [16]. Researchers have also achieved the same impressive results in bird identification and skin cancer diagnosis.

* Games were once considered the last bastion of human intelligence. Since the beginning of the use of time-differentiated reinforcement learning to play Backgammon's TD-Gammon, the development of algorithms and computing power has spawned a series of new algorithms used in games. Unlike Backgammon, chess has more complex state space and more optional actions. "Deep Blue" defeated Garry Kasparov with a large number of efficient searches of parallel, dedicated hardware and game trees [17]. Go is considered a more difficult game because of its huge state space. In 2016, AlphaGo reached the human level with a combination of deep learning and Monte Carlo tree sampling [18]. For Texas hold'em, in addition to the huge state space, the bigger challenge is that the game's information is not completely visible, such as not seeing the opponent's cards. The "Libratus" has surpassed the performance of human players with an efficient strategy system [19]. The above examples all show that advanced algorithms are an important reason for the improvement of artificial intelligence in games.

* Another sign of machine learning progress is the development of autonomous vehicles. Although there is still a long way to go before full autonomous driving, products with partial autonomous driving functions from companies such as [Momenta](http://www.momenta.com), [Tesla](http://www.tesla.com), [Nvidia](http://www.nvidia.com), [MobilEye](http://www.mobileye.com) and [Waymo](http://www.waymo.com) have shown great progress in this area. The difficulty of completely autonomous driving is that it requires the integration of perception, thinking and rules in the same system. At present, deep learning is mainly applied to the part of computer vision, and the rest of it still requires a lot of debugging by engineers.

The above list is just the tip of the iceberg of the results of deep learning in recent years. Some of the developments in robotics, logistics management, computational biology, particle physics, and astronomy have been attributed to deep learning in recent years. It can be seen that deep learning has gradually evolved into a universal tool that can be used by both engineers and scientists.


## Features

Before describing the features of deep learning, let's review and summarize the relationship between machine learning and deep learning. Machine learning studies how computer systems can use experience to improve performance. It is a branch of artificial intelligence and a means of implementing artificial intelligence. In many research directions in machine learning, representational learning focuses on how to automatically find the appropriate way to represent data in order to better transform the input into the correct output. The deep learning that this book focuses on is a representational learning method with multiple levels of representation. At each level (starting with the original data), deep learning transforms the representation of the current level into a higher level representation by a simple function. Therefore, the deep learning model can also be seen as a function that is composed of many simple functions. When these composite functions are sufficient, the deep learning model can express very complex transformations.

Deep learning can represent increasingly abstract concepts or patterns step by step. Take an image as an example, its input is a bunch of raw pixel values. In the deep learning model, images can be expressed step by step as edges of specific positions and angles, patterns obtained by combination of edges, patterns of specific parts obtained by further combining a plurality of patterns, and the like. Ultimately, the model can easily accomplish a given task based on a higher level representation, such as identifying objects in an image. It is worth mentioning that as a kind of representational learning, deep learning will automatically find the appropriate way to represent data at each level.

Therefore, an external feature of deep learning is end-to-end training. In other words, instead of assembling the parts that are individually debugged together to form a system, the whole system is built and then trained together. For example, computer vision scientists once separated the feature structures from the construction of machine learning models. And things like Canny edge detection[20] and SIFT feature extraction[21] had dominated for more than 10 years, but they were the best way that humans could find. When deep learning entered this field, these feature extraction methods were replaced by more powerful, automatically optimized step-by-step filters.

Similarly, in the field of natural language processing, the bag-of-words model had been considered the best choice for many years [22]. The bag-of-words model is one that maps a sentence to a word frequency vector, but this approach completely ignores the order of the words or the punctuation in the sentence. Unfortunately, we also had no ability to manually build better features. But automated algorithms can instead search for the best of all possible feature designs, which has also brought great progress. For example, semantically related word embedding can do the following reasoning in vector space: "Berlin - Germany + China = Beijing". It can be seen that these are the effects of end-to-end training of the entire system.

In addition to end-to-end training, we are also moving from a statistical description with parameters to a model with no parameters at all. When the data is very scarce, we need to get a practical model by simplifying the assumptions about reality. When the data is sufficient, we can replace these parametric models with a non-parametric model that better fits the reality. This also allows us to get a more accurate model, although we need to sacrifice some interpretability.

Another difference between deep learning and previous work is the inclusion of non-optimal solutions, the use of non-convex nonlinear optimization, and the audacity to try methods that have not been proved. This new wave of empiricism in dealing with statistical issues and the influx of a large number of talents have brought about rapid progress on practical issues, although in most cases it is necessary to modify or even reinvent the tools that have existed for decades.

Finally, the deep learning community has long prided in sharing tools between academics and businesses, and has open sourced many excellent software libraries, statistical models, and pre-training networks. It is in the spirit of open source and openness that this book and the teaching videos based upon it can be freely downloaded and shared. We are committed to lowering the threshold for deep learning for everyone and hope that everyone will benefit from it.


## Summary

* Machine learning studies how computer systems can use experience to improve performance. It is a branch of artificial intelligence and a means of implementing artificial intelligence.
* As a class of machine learning, representational learning focuses on how to automatically find the appropriate way to represent data.
* Deep learning is a representational learning method with multiple levels of representation. It can represent increasingly abstract concepts or patterns step by step.
* The core idea of using data programming and neural networks, upon which deep learning is based, has been studied for hundreds of years.
* Deep learning has gradually evolved into a universal tool that can be used by both engineers and scientists.


## exercise

* Is there any part of the code you are writing that can be "learned"? In other words, is there a part that can be improved by machine learning?
* Have you ever encountered some problem in your life that has many examples, but no specific auto-solving algorithm can be found? They may be the best prey for deep learning.
* If the development of artificial intelligence is regarded as a new industrial revolution, then is the relationship between deep learning and data like the relationship between steam engines and coal? Why? 
* Where else can the end-to-end training method be used? Physics? Engineering? Or economics?
* Why should we let the deep network imitate the human brain structure? Why should we not let the deep network imitate the human brain structure?

## Scan the QR code to get to the [forum](https://discuss.gluon.ai/t/topic/746)

![](../img/qr_deep-learning-intro.svg)

## References

[1] Machinery, C. (1950). Computing machinery and intelligence-AM Turing. Mind, 59(236), 433.

[2] Hebb, D. O. (1949). The organization of behavior; a neuropsychological theory. A Wiley Book in Clinical Psychology. 62-78.

[3] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: a simple way to prevent neural networks from overfitting. The Journal of Machine Learning Research, 15(1), 1929-1958.

[4] Bishop, C. M. (1995). Training with noise is equivalent to Tikhonov regularization. Neural computation, 7(1), 108-116.

[5] Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473.

[6] Sukhbaatar, S., Weston, J., & Fergus, R. (2015). End-to-end memory networks. In Advances in neural information processing systems (pp. 2440-2448).

[7] Reed, S., & De Freitas, N. (2015). Neural programmer-interpreters. arXiv preprint arXiv:1511.06279.

[8] Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., … & Bengio, Y. (2014). Generative adversarial nets. In Advances in neural information processing systems (pp. 2672-2680).

[9] Zhu, J. Y., Park, T., Isola, P., & Efros, A. A. (2017). Unpaired image-to-image translation using cycle-consistent adversarial networks. arXiv preprint.

[10] Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2017). Progressive growing of gans for improved quality, stability, and variation. arXiv preprint arXiv:1710.10196.

[11] Li, M. (2017). Scaling Distributed Machine Learning with System and Algorithm Co-design (Doctoral dissertation, PhD thesis, Intel).

[12] You, Y., Gitman, I., & Ginsburg, B. Large batch training of convolutional networks. ArXiv e-prints.

[13] Jia, X., Song, S., He, W., Wang, Y., Rong, H., Zhou, F., … & Chen, T. (2018). Highly Scalable Deep Learning Training System with Mixed-Precision: Training ImageNet in Four Minutes. arXiv preprint arXiv:1807.11205.

[14] Xiong, W., Droppo, J., Huang, X., Seide, F., Seltzer, M., Stolcke, A., … & Zweig, G. (2017, March). The Microsoft 2016 conversational speech recognition system. In Acoustics, Speech and Signal Processing (ICASSP), 2017 IEEE International Conference on (pp. 5255-5259). IEEE.

[15] Lin, Y., Lv, F., Zhu, S., Yang, M., Cour, T., Yu, K., … & Huang, T. (2010). Imagenet classification: fast descriptor coding and large-scale svm training. Large scale visual recognition challenge.

[16] Hu, J., Shen, L., & Sun, G. (2017). Squeeze-and-excitation networks. arXiv preprint arXiv:1709.01507, 7.

[17] Campbell, M., Hoane Jr, A. J., & Hsu, F. H. (2002). Deep blue. Artificial intelligence, 134(1-2), 57-83.

[18] Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., Van Den Driessche, G., … & Dieleman, S. (2016). Mastering the game of Go with deep neural networks and tree search. nature, 529(7587), 484.

[19] Brown, N., & Sandholm, T. (2017, August). Libratus: The superhuman ai for no-limit poker. In Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence.

[20] Canny, J. (1986). A computational approach to edge detection. IEEE Transactions on pattern analysis and machine intelligence, (6), 679-698.

[21] Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. International journal of computer vision, 60(2), 91-110.

[22] Salton, G., & McGill, M. J. (1986). Introduction to modern information retrieval.
