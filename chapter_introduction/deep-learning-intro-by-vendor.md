# Introduction to Deep Learning

You might already have some experience with programming and may have already developed a few programs of your own.  You may also have read those ubiquitous reports about deep learning or machine learning. Oftentimes these terms are given a broader name: artificial intelligence.  Fortunately, the fact is that most programs do not require deep learning or artificial intelligence techniques to function.  For example, if we were to code a microwave oven user interface, we could design dozens of buttons and a set of rules that accurately describes the microwave oven’s performance under various conditions with little effort.  In the case of something like an email client, a program that is more complex than a microwave oven interface, we can still utilize a step by step thought process: the client user interface will need several input boxes to accept details like recipients, message subjects, and email body, etc.. The email client will detect keyboard inputs and write these into a buffer, displaying them in the corresponding input boxes.  When the user clicks the “Send” button, the email client will be tasked with checking things like whether the recipients’ email address is in the correct format and whether the email subject field has been left empty, warning the user if it has and then making use of the corresponding protocol to send the email. 

It is worth noting that in the above examples, we would not need to collect real world data, nor would we need to systematically extract any features from this data.  Our common sense and programming skills are enough to complete these tasks, as long as there is enough time available. 

As easy as it is to code such simple programs, it’s also easy to find simple queries that even the world’s best programmers can’t solve using just their programming skills.  For example, suppose we want to code a program that determines if a cat is present in a given image. Sounds simple, right? All the program needs to do is output “true” in the case of a cat being present or “false” if a cat is not present in each image provided.  Surprisingly, even the world’s best computer scientists and programmers aren’t able to code such a program.

So, where should we start? Let’s start by simplifying this problem. If we assume that all of the input images have the same height and width of 400 pixels, and one pixel consists of three values (red, green and blue), then each image is represented by close to 500,000 numerical values.  So which values hide the necessary information? Is it the average of all values, the values of the four corners, or the values of a particular point in the image? In fact, in order interpret an image’s content and determine whether it contains a cat, you need to look for features like edges, textures, shapes, eyes, noses, etc. Such features only appear when thousands of values are combined. 

One way to solve the above problem is to think in reverse. Rather than coding a problem-solving program, we might as well analyze the final requirement to find a solution.  This is also in fact the core idea shared between the latest machine and deep learning applications. We call this “programming with data.” Rather than sit in a room and think about how to code a program to recognize a cat, it would make more sense to make use of a human’s ability to recognize a cat in an image.   We can start by collecting a mixed set of real images, with some containing a cat and some not without. Next, our goal will be to figure out how we can use these images to define a function that is able to infer whether a given image contains a cat or not.  The function’s form is usually based on our knowledge about a specific problem. For example, we can use a quadratic function to determine whether the given image contains a cat or not.  However, the specific value of a function parameter., such as in the case of the coefficient value of a quadratic function ,is determined by data.

Machine learning is generally a discipline that covers a wide variety of functional forms that can be applied to different problems, as well as the efficient usage of data in order to obtain specific function parameter values.  On the other hand, deep learning typically refers to a class of functions in machine learning, usually in the form of multi-layer neural networks.  Recently, thanks to increasingly powerful hardware and a reliance on large data sets, deep learning has gradually become the go-to method for processing complex, high-dimensional data like images, text corpora and sound signals. 

We have now entered an era of programming that is benefitting from more and more assistance from deep learning.  It can be said that we are at a turning point in the history of computer science.  Deep learning is already used in everyday objects, for example it can be found in your phone in the form of spelling correction, text prediction, speech recognition, facial recognition in photographs, and more.  Nowadays most software engineers have the ability to build complex models to solve problems that even the best scientists found challenging ten years ago. This has all been made possible thanks to excellent algorithms, fast and inexpensive computing power, unprecedented amounts of data, and powerful software tools. 

The aims of this book are to help you enter the world of deep learning.  We hope that through the combination of math, codes and examples that you’ll be able to harness the knowledge of deep learning at your fingertips.  This book does not require you to have a deep background in either mathematics or programming, and we will be explaining the necessary knowledge step by step throughout the chapters.  It is worth mentioning that each section of the book is a Jupyter notebook that can be independently run. You can find these notebooks online and run them from your PC or cloud server.  This allows you to change any code in the book and receive timely feedback.  Through this book, we hope to inspire and help a new generation of programmers, entrepreneurs, statisticians, biologists, and anyone interested in problem solving through deep learning. 


## Origin

Although the term “deep learning” seems to have emerged within recent years, the basis of deep learning, that is the use data programming along with neural network, has been studied for hundreds of years  Humans have been eagerly analyzing data in order to predict the future since ancient times.  In fact, data analysis is the essence of most natural sciences. Humans have been extracting rules from everyday observations to look for uncertainties for centuries.

As early as the 17th century, [Jacob Bernoulli (1655-1705)](https://en.wikipedia.org/wiki/Jacob_Bernoulli)  had had described a stochastic process with only two possible outcomes (such as a coin flip). This came to be known as the Bernoulli Distribution. About a century later, [Carl Friedrich Gauss (1777-1855)](https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss) was the first to define the least squares method, a method that is still widely used today in various fields ranging from insurance calculation to medical diagnosis. Tools such as probability theory, statistics, and pattern recognition helped natural science experimenters regress from data to natural laws, discovering a series of natural laws that are perfectly expressed by linear models, like Ohm's law (the law describing the relationship between the voltage across a resistor and the current flowing through the resistor).

Even in the Middle Ages, mathematicians were keen to make use of statistics when calculating estimates. For example, [Jacob Kobels' (1460-1533)](https://www.maa.org/press/periodicals/convergence/mathematical-treasures-jacob-kobels-geometry) geometric book documented using the average length of 16 men’s feet to determine the average length of a single man’s foot. 

![In the Middle Ages, the average length of 16 men’s feet to determine the average length of a single man’s foot.. ](../img/koebel.jpg)

As shown in Figure 1.1, in this study, when leaving church 16 adult men were asked to stand in a row with their feet together, and then the total length of their feet was divided by 16 to determine an estimated average. Today, this number represents approximately one foot. This algorithm was later modified to take into consideration a foot’s specific shape: the longest and the shortest feet were not considered, and an average was only taken of the remaining feet, i.e., an early example of the trimmed mean.

The real take-off in 20th century modern statistics was attributable to the collection and distribution of data. [Ronald Fisher (1890-1962)](https://en.wikipedia.org/wiki/Ronald_Fisher), the father of modern statistics, contributed hugely to the statistical theory and the application of statistics in genetics. Many of the algorithms and formulas he invented, such as linear discriminant analysis and the Fisher information, are still frequently used. Even the Iris data set he published in 1936 is occasionally used to demonstrate machine learning algorithms.

The information theory defined by [Claude Shannon (1916-2001)](https://en.wikipedia.org/wiki/Claude_Shannon) and the computational theory by [Allan Turing (1912-1954)](https://en.wikipedia.org/wiki/Allan_Turing) have also had a profound impact on machine learning. In his famous paper [Computing Machinery and Intelligence](https://www.jstor.org/stable/2251299) , turning put forward a question;  "Can machines think?"[1]. In the "Turing Test" he described that when using a textual interaction, if a person can't distinguish whether his conversation object is human or machine, then the machine can be considered intelligent. Nowadays, the development of artificial intelligence is changing with each passing day. 

Other areas that have had a major impact on deep learning include neuroscience and psychology.  Since human beings can clearly demonstrate intelligence, it is reasonable that we explore the explanation behind this and reverse engineer the human intelligence mechanism.  One of the earliest algorithms was formally proposed by [Donald Hebb (1904-1985)](https://en.wikipedia.org/wiki/Donald_O._Hebb). In his groundbreaking book, [The Organization of Behavior](http://s-f-walker.org.uk/pubsebooks/pdfs/The_Organization_of_Behavior-Donald_O._Hebb.pdf), Hebb suggested that nerves learn through positive reinforcement. Nowadays this is known as the Hebb Theory[2]. The Hebb Theory is an early example of the perceptron learning algorithm and has become the cornerstone of many stochastic gradient descent algorithms that support today's deep learning: reinforcing desired behavior, punishing undesired behavior, and finally, obtaining excellent neural network parameters.

The name Neural Network was inspired by human biology.  Such researchers like [Alexander Bain (1818-1903)](https://en.wikipedia.org/wiki/Alexander_Bain) and [Charles Scott Sherrington (1857-1952)](https://en.wikipedia.org/wiki/Charles_Scott_Sherrington) can be traced back to over a century ago. Researchers have since attempted to build computational circuits that mimic neuronal interactions. Over time, the neural network’s biological interpretation has been diluted, but the name has been retained. Today, most neural networks consist of the following core principles:

* The alternating use of linear and nonlinear processing units, often referred to as "layers". 
* The use of the chain rule (i.e. back-propagation) to update the network parameters.

From around 1995 to 2005, after machine learning’s initial rapid development, most of the researchers’ attention moved away from the neural network.  This is due to various reasons. First of all, the training of neural networks requires a great deal of computing power.  Although the memory was sufficient at the end of the last century, the computing power was not. Secondly, the data set used at the time was also relatively small. Released in 1932, Fisher’s Iris data set which was widely used to test the performance of algorithms consisted of only 150 examples.  Although now considered to be a typical simple data set, at the time the MNIST data set, with 60,000 examples, was considered to be very large.  Empirically, due to the scarcity of data and computing power at the time, tools such as the kernel method, the decision tree and the probabilistic graphical model were superior.  Unlike neural networks, these tools do not require long-term training and are able to provide predictable results under strong theoretical guarantees. 

## Development

With the rise of the Internet as well as cost-effective sensors and low-cost memory, it is now easier for us to access large amounts of data.  Coupled with the cheap computing power, in particular the GPUs originally designed for computer games, the past situations described in the past have since changed drastically.  Now, algorithms and models that were originally thought impossible are now available at your fingertips in an instant.  The following table provides evidence of this trend:

|Year|Number of Data Examples|Memory|Floating Point Calculations per Second|
|:--|:-:|:-:|:-:|
|1970|100 (Iris)|1 KB|100 K (Intel 8080)|
|1980|1 K (Boston house prices)|100 KB|1 M (Intel 80186)|
|1990|10 K (handwritten character recognition)|10 MB|10 M (Intel 80486)|
|2000|10 M (web page)|100 MB|1 G (Intel Core)|
|2010|10 G (advertising)|1 GB|1 T (NVidia C2050)|
|2020|1 T (social network)|100 GB|1 P (Nvidia DGX-2)|

Clearly, storage capacity has not managed to keep up with the growth in data volume.  At the same time though, the increase in data volume has been overshadowed by the increase in computing power available.  This trend has allowed statistical models to invest more computational power into parameter optimization, but at the same time there is a need to improve the efficiency of storage utilization. One of the ways this can be done is by using nonlinear elements.  This has led to machine learning and  statistics becoming the preferred choice in the development of multilayer neural networks, replacing generalized linear models and kernel methods.  Such changes are the reasons for the “re-discovery” of the pillar models of deep learning, such as multilayer perceptrons, convolutional neural networks, long- and short-term memory-cycle neural networks, and Q-learning, throughout the past decade after years of being disregarded.

Recent advances in statistical models, applications, and algorithms have often been compared to the Cambrian explosion (an important point in history during which the number of historical animal species exploded). However, these advancements have not come around simply because we now have more resources available to allow old content to be presented in a new form.  The following list covers just a few of the reasons behind the significant development of deep learning in the past decade: 

* Exceptional capacity control methods, such as dropout, mean that the training of large networks is no longer subject to overfitting (the behavior of a large neural network learning to memorize most training data) [3]. This is done by injecting noise into the entire network, for example by randomly assigning the weight as a random number during training [4].

* The attention mechanism solves another problem that had plagued statisticians for more than a century: the expansion of memory capacity and system complexity without the need for additional parameters.  Through the use of a learnable pointer, the attention mechanism is able to construct a complex solution. [5]. That is to say that it is better to memorize the pointer using the intermediate translation state rather than memorizing the entire sentence in a task, as is the case in machine translation.  Such a structure makes it possible to accurately translate long sentences, since there is no need for the information contained in the original text to be stored before the translation is generated. 

* Multilevel designs, such as memory networks [6] and neural encoder-interpreters [7], make iterative modeling methods for inference processes possible. These models allow you to modify the deep network’s internal state, making it possible to simulate the steps on the interface chain in the same way as if the processor was modifying memory during the computation. 

* The discovery of the generative adversarial network was another major development [8]. Traditionally, the statistical methods used in probability distribution estimation and generative models were more focused  on determining the correct probability distribution and sampling algorithm. A key innovation in the generative adversarial network is the replacement of the sampled part with  [an arbitrary] algorithm containing differentiable parameters. In turn, these parameters will be trained until the discriminator to no longer be able to distinguish between real and generated examples. The feature of generative adversarial network using arbitrary algorithms to generate results allows for many new opportunities []. The examples of the generation of running zebras[9] and the generation of celebrity photos[10] are both evidence of the development of the generative adversarial network.

* A single GPU is no longer sufficient in the majority of cases involving training on large data sets.  Over the past decade, our ability to build distributed parallel training algorithms has dramatically improved.  The biggest bottleneck in the design of scalable algorithms lies at the heart of the deep learning optimization algorithm: stochastic gradient descent requiring relatively smaller batches. However, at the same time, smaller batch sizes, will also reduce the GPU’s efficiency.  If we utilize 1024 GPUs and each GPU is allocated a batch size of 32 examples, the single-step training will have a batch size of more than 32,000.   Recently, the work of Li Mu [11], Yang You, et al [12], and Xianyan Jia, et al [13], has increased the batch size to as many as 64,000 examples and reduced the ResNet-50 model’s training time on the ImageNet data set to 7 minutes. Previously, the initial training time was calculated in days.

* The implementation of parallel computing also has contributed to the development of reinforcement learning where the simulation can at least be utilized. Parallel computing has helped computers perform at a level higher than that of humans in Go, Atari games, StarCraft, and physical simulation.

* The deep learning framework also plays an important role in the process of disseminating deep learning ideas. For example, first generation frameworks such as [Caffe](https://github.com/BVLC/caffe), [Torch](https://github.com/torch) and [Theano](> simplified modeling. These frameworks have since been utilized in many groundbreaking papers. They have since been replaced by [TensorFlow](https://github.com/tensorflow/tensorflow) (often used in the form of high level API [Keras](https://github.com/keras-team/keras)), [CNTK](https://github.com/Microsoft/CNTK), [Caffe 2](https://github.com/caffe2/caffe2) and [Apache MXNet](https://github.com/apache/incubator-mxnet). The third generation framework, i.e. the imperative deep learning framework, was pioneered by [Chainer](https://github.com/chainer/chainer) which defined the model with syntax similar to NumPy. This idea was later adopted by [PyTorch](https://github.com/pytorch/pytorch) and MXNet's [Gluon API](https://github.com/apache/incubator-mxnet), with the latter being the tool used by this book to teach deep learning.

System researchers are responsible for building better tools and statisticians are responsible for building better models. This division of labor greatly simplifies work. For example, the  training of a logistic regression model was an assignment assigned to new doctoral students studying machine learning at Carnegie Mellon University in 2014. Today, this problem can be solved using less than 10 lines of code, and is a task which can be carried out by ordinary programmers.

## Successful Cases

Throughout the years, machine learning has achieved goals that were previously unattainable using other methods.  As an example, mail sorting has been using optical character recognition since the 1990s.  In fact, this [machine learning?] is the source of the well-known handwritten MNIST and USPS digital data sets. Machine learning is also the backbone of the electronic payment systems used to read bank checks, perform credit ratings, and prevent financial fraud. Machine learning algorithms are also used on the web to provide search engine results, personalized recommendations, and to sort web pages. Machine learning has made its way into every aspect of our work and everyday life, despite the fact that it has only been known to the public for a short time.  Recently, machine learning has gradually become the public’s main focus as a result of breakthroughs in consumer related problems that were once considered unsolvable.  These advancements are largely attributed to deep learning:

* Intelligent assistants like Apple's Siri, Amazon's Alexa, and Google Assistant are able to answer verbal commands, ranging from simple functions such as turning on/off the light (which helps disabled individuals significantly) to assisting in voice conversations, with considerable accuracy. The emergence of intelligent digital assistants may serve as a sign that artificial intelligence is beginning to influence our everyday lives.

* The key requirement for intelligent assistants is the ability to accurately recognize speech, and the accuracy of  the speech recognition in some systems has gradually grown to be very close to human ability[14].

* Object recognition has also undergone a long development process. In 2010, the identification of object categories in images was still a challenging task.  In the same year, a top-5error rate of 28% was achieved in the ImageNet baseline test [15]. By 2017, this number had dropped to 2.25% [16]. Researchers have also achieved similarly impressive results in the fields of bird identification and skin cancer diagnosis.

* Games were once considered to be the last bastion of human intelligence. Since the early stages of the use of time-differentiated reinforcement learning to play Backgammon's TD-Gammon, algorithmic and computing power developments have spawned a series of new algorithms used in such games. Unlike Backgammon, chess has a more complex state space and more optional actions. Thanks to a large number of efficient parallel searches, dedicated hardware and game trees, the chess computer "Deep Blue" was able to defeat chess grandmaster Garry Kasparov [17]. Because of its huge state space, “Go” is considered to be a more difficult game because. In 2016, using a combination of deep learning and Monte Carlo tree sampling, the program “AlphaGo” was able to reach a level of playing similar to that of humans. [18]. In Texas hold'em the game is not completely visible, for example you cannot see your opponent’s cards, which poses a big challenge, in addition to the huge state space. The artificial intelligence program "Libratus" has surpassed the performance of human poker players through the use of an efficient strategy system [19]. All of the above examples prove have been and will always be an important factor in the improvement of artificial intelligence in games. 

* Another sign of machine learning progress is the development of autonomous vehicles. Although there is still a long way to go before fully autonomous vehicles reach consumers, vehicles with partial driving autonomy from companies such as [Momenta](http://www.momenta.com), [Tesla](http://www.tesla.com), [Nvidia](http://www.nvidia.com), [MobilEye](http://www.mobileye.com) and [Waymo](http://www.waymo.com) have shown significant progress in this field.  The main challenge preventing completely self-driven vehicles is the need to have perception, thinking and rules integrated into one system.  Currently, deep learning is mainly applied in computer technology, and its application in other fields still requires significant debugging by engineers.

The above list of deep learning achievements in recent years is just the tip of the iceberg.  Some developments made in recent years in the fields of robotics, logistics management, computational biology, particle physics, and astronomy have also been attributed to deep learning. It can also be noted that that deep learning has gradually evolved into a universal tool used by both engineers and scientists.


## Features

Before we look into the features of deep learning, let's review and summarize the relationship between machine learning and deep learning. Machine learning studies how computer systems can improve their performance based on experience.  It is a branch of artificial intelligence and a means of implementing such. In many research branches of machine learning, representational learning focuses on how to automatically find the most appropriate way to represent data in order to better transform the input into the correct output. The type of deep learning focused on in this book is a representational learning method with multiple levels of representation. At each level (beginning with the original data), deep learning transforms the current level’s representation into a higher level representation using a simple function. Therefore, the deep learning model can also be seen as a complex function composed of many simple functions. When enough of these composite functions are present, the deep learning model is able to express very complex transformations.

Deep learning is able to gradually represent increasingly abstract concepts or patterns. Taking an image as an example, its input is a collection of raw pixel values. In the deep learning model, images can be gradually expressed as edges of specific positions and angles, patterns obtained by a combination of edges, patterns of specific parts obtained by the further combining a plurality of patterns, and so on, so forth. Based on a higher level representation, the model is able to easily accomplish a given task, such as the identification of particular objects in an image.  It is worth noting that as a kind of representational learning, deep learning will automatically find the appropriate way to represent data at each level.

Therefore, end-to-end training is an external feature of deep learning. In other words, the entire system is built and trained together rather than being formed by joining individually debugged parts.  For example, computer vision scientists had previously separated feature structures from machine learning models such as the Canny edge detection[20] and SIFT feature extraction[21], which had dominated for over 10 years which were the best methods available at the time.  With the introduction of deep learning, these feature extraction methods were replaced by more powerful, automatically optimized gradual filters. 

Similarly, in the field of natural language processing, the bag-of-words model had for many years been considered the best choice [22]. The bag-of-words model is one that maps a sentence to a word frequency vector, completely ignoring word order and punctuation in the sentence. Unfortunately, at the time we did not have the ability to build better features. However, nowadays automated algorithms can search for the best possible feature designs, which has also brought about great progress.  For example, semantically related word embedding can perform the following reasoning in vector space: "Berlin - Germany + China = Beijing". It can be seen that these are the effects of end-to-end training throughout the entire system.

In addition to end-to-end training, we are also moving from statistical models using parameters to models with no parameters whatsoever.  When data is very scarce, we need to obtain practical model by simplifying any assumptions made about reality. When we have sufficient data, we can replace these parametric models with non-parametric models that better fit the reality.  This also allows us to achieve more accurate model, although some interpretability does need to be sacrificed. 

Additional differences between deep learning and previous work include: the inclusion of non-optimal solutions, the use of non-convex nonlinear optimization, and the unsuccessful attempts using unproven methods. This new wave of empiricism in dealing with statistical issues along with the influx of rising talent has brought about rapid progress in practical issues; although in most cases it is necessary to modify or even reinvent tools that have existed for decades.

Finally, the deep learning community has long prided itself in the sharing of resources between academics and businesses, and has open sourced many excellent software libraries, statistical models, and pre-training networks. It is in the spirit of open source and openness that this book and the teaching videos based upon it can be freely downloaded and shared. We are committed to lowering the threshold for deep learning for everyone and hope that everyone will benefit from it.


## Summary

* Machine learning studies how computer systems can improve performance based on experience. It is a branch of artificial intelligence and a means of implementing such.
* As a class of machine learning, representational learning focuses on automatically finding the appropriate way to represent data.
* Deep learning is a representational learning method consisting of multiple levels of representation. It can gradually represent increasingly abstract concepts or patterns.
* The core idea upon which deep learning is based, the use of data programming and neural networks, has been studied for centuries.
* Deep learning has gradually evolved into a universal tool that can be used by both engineers and scientists.


## exercise

* Is there any part of the code that you are writing that can be "learned"? In other words, is there a part that can be improved by machine learning?
* Have you ever encountered a problem in your life that has many examples, but cannot find any specific auto-solving algorithm? They may be the best targets for deep learning.
* If the development of artificial intelligence is regarded as a new industrial revolution, then is the relationship between deep learning and data similar to the relationship between steam engines and coal? Why? 
* Where else can the gradual training method be implemented? Physics? Engineering? Or economics?
* Why should we let the deep network imitate the human brain structure? Why should we not let the deep network imitate the human brain structure?

## Scan the QR code to access [forum](https://discuss.gluon.ai/t/topic/746)

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
