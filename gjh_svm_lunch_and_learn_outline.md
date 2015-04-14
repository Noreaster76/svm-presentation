## introduction
very brief mention of what SVMs are: a popular, easy-to-use machine
learning algorithm

## plan of attack
outline the content of today's talk:
1. demo of SVM running on some example data
2. interpreting the results
3. how the heck did that happen?
4. review of fundamental concepts of machine learning and why machine
   learning is helpful
5. explanation of how SVMs work
6. explanation of how to train SVMs
7. some classic examples of where SVMs are currently used
8. some publically available SVM implementations

## demo
i will train an SVM on some example data, say, [the heart dataset](http://archive.ics.uci.edu/ml/datasets/Statlog+%28Heart%29)

## run the trained model against a test set
calculate the test performance of the SVM

## draw a contrast between the test set results and the baseline case
machine learning algorithms earn their keep by finding patterns in raw
data (well, raw data that has been encoded in a way that the machine
learning algorithm can ingest, and also, in many cases, has also been
massaged to remove data deemed irrelevant or to normalize data with high
variance)

## machine learning review
from _machine learning in action_, by peter harrington:
```
machine learning uses statistics. in engineering, we're used to solving
a deterministic problem where our solution solves the problem all the
time. if we're asked to write software to control a vending machine, it
had better work all the time, regardless of the money entered or the
buttons pressed. there are many problems where the solution isn't
deterministic. that is, we don't know enough about the problem or don't
have enough computing power to properly model the problem. for these
problems we need statistics.
```
### attribute
explain what an attribute is, using the heart dataset as an example
### classification
explain what classification is, using the heart dataset as an example;
note that today, we are just going to focus on supervised learning for
classification problems, but that other important disciplines of machine
learning exist, such as linear regression, which is supervised, and
clustering, which is unsupervised
### supervised vs. unsupervised learning
### training data
explain what a training data set is, using the heart dataset as an example.
### target variable
the class of each training or test example. as far as the example of the
heart dataset goes, the target variable, what we're trying to find a
pattern in and classify, is whether heart disease is present.
from harrington: `the machine learns by finding a relationship between
the [training data points] and the target variable`
### validation data
explain what a validation data set is, using the heart dataset as an example
### knowledge representation
can a human interpret the patterns that the machine has found, or is the
model an opaque black box?

## how to train and use a machine learning model
from harrington:
1. collect a data set
2. encode the data set in such a way that the machine learning algorithm
   will be able to ingest
3. filter out any known noise or other useless attributes or rows in the
   data, so that the machine learning algorithm doesn't get distracted
by something you know to be irrelevant
4. train the algorithm. `you feed the algorithm good clean data from the
   first two steps and extract knowledge or information`. the
"knowledge" or "information" *is* your model.
5. you then exercise your model on the validation set, the subset of the
   training data that you've kept out of training the model. this will
show how well your model performs with unknown data. has it overfit the
training data, or is it a model that reflects a truly general pattern in
the data and will perform well in the future when supplied with data
points that have some variance in their attribute values?
6. use your model in production. you may want to periodically re-train
   your model, as you collect more data in your system.

### overfitting
show example from [scikit-learn](http://scikit-learn.org/stable/auto_examples/plot_underfitting_overfitting.html)

## how SVMs work
### linearly separable data with maximum margin
use examples at [scikit-learn.org](http://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html#example-svm-plot-separating-hyperplane-py) and the visualizer at [the LibSVM page](http://www.csie.ntu.edu.tw/~cjlin/libsvm/index.html?js=1#svm-toy-js) to demonstrate separating the data with the greatest possible margin
### what if the data are not linearly separable?
discuss slack variables, which allow us some room for error and find a
line that doesn't necessarily separate the data perfectly, but still
does the best job we can.
### what if the data are _really_ not linearly separable?
discuss higher-order SVMs, such as polynomial and Gaussian radial basis
function SVMs.
- if you have 2 attributes of data, then it's very much is like you're
  trying to position a yardstick in such a way that it will best
separate two 2-D clouds of data points.
- if you have 3 attributes of data, then it's as if you're trying to
  position a flat, infinite plane (this is known as _the separating
hyperplane_) in such a way that it will best
separate two 3-D clouds of data points.
- the metaphor continues with 4 attributes of data (finding the best
  separating hyperplane in 4 dimensions), 5 attributes, and so on.
#### HOWEVER
if you can't find a linear or flat hyperplane that separates the two
classes of data (whether you're allowing for some points to fall on the
wrong side of the line/plane or not), what if you were to _bend_ the
line or plane? like, what if, instead of trying to fit a straight line
or a flat plane to separate the data, you just BENT THE LINE?
- there is no spoon
- keanu whoa
- example of fitting a parabola
- example of fitting a plane to separate a disk of data points from an
  outer circle of data points that surrounds it: show [this video](https://youtu.be/3liCbRZPrZA)
### grid search for finding the best performing value of C and gamma
we try to home in on the best performing combinations of C (the cost
parameter, or how willing we are to settle for a margin that doesn't
perfectly separate the data points in the training set) and gamma
### SMO?
discuss the sequential minimal optimization algorithm? i'm not familiar
with it yet and would have to learn it

## advantages of SVMs
- according to harrington, they're widely considered to be the best
"stock" classifier: you can take the SVM algorithm right off the shelf,
and train a learner on your
data and get very good or excellent performance on it. witness the
performance of SVMs on analyzing handwritten digits: an SVM with a
relatively simple configuration correctly identified a handwritten digit
~98.6% of the time, whereas a top-of-the-line artificial neural network
delivered ~99.75% accuracy. the artificial neural network therefore made
1/6 the number of errors that the SVM did, but still, that ~98.6% figure
sure isn't bad for an off-the-shelf algorithm. [see the MNIST page on
handwritten digits for comparison of learner performance across a good
variety of learners](http://yann.lecun.com/exdb/mnist/)
- in terms of configuration, they are relatively very simple and require very little tuning of parameters

## disadvantages of SVMs
- black box. as opposed to some other machine learning algorithms, such
  as decision trees, the various weights an SVM assigns to the attributes in
  your data are not something you can take away and draw conclusions
from. you can't extract knowledge on, say, which attributes of the heart
dataset are most
highly correlated with risk of heart disease so that you can go issue
public health advisories, or which attributes of your sales data
highlight the best sales leads so that you can advise your marketing
team accordingly. so, it depends on whether you just want an answer
(classify an unknown data point into a certain category) or, instead,
you want greater understanding from your data.
** this may not be completely true. see [colin campbell's lecture](http://videolectures.net/epsrcws08_campbell_isvm/), in which he says you can interpret the results based on the support vectors. maybe it's possible to interpret the results from an SVM, it's just a bit more difficult than with a decision tree, for example. **
- can be slow. according to [scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html), the time to train a kernel (meaning, higher dimension) SVM grows by more than O(n^2) (where n is the number of training examples), so, much beyond 10,000 examples, you're looking at a long time to train. however, with a linear SVM (one where you're just trying to linearly separate data, like with a yardstick), the performance is O(n). according to [yoshua bengio of the university of montreal](https://www.quora.com/Why-is-kernelized-SVM-much-slower-than-linear-SVM):
```
when the number of training examples is large (e.g. millions of documents, images, or customer records), kernel SVMs are too expensive for training and very expensive for classification
```
(thanks to Data Will for pointing me toward that content)

## popular real-world applications of SVMs
from wikipedia:
- text categorization
- image classification
- protein classification in medical research
- hand-written character recognition (perhaps this is something we might
  have a need for someday)

## popular SVM implementations
- [LibSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/index.html)
- [SVMLight](http://svmlight.joachims.org/)
- [scikit-learn](http://scikit-learn.org/)
### not very difficult to use in your own code
using an SVM from within your code is actually not very difficult, since
these libraries have somewhat clear APIs.
from data will, in python, using scikit-learn
```python
from sklearn.svm import SVC

X = my training data (the stuff we just graphed, with 2 columns - blood pressure and age - for each observation)
y = my training labels (the colors of the dots)

clf = SVC(C=1.0)
clf.fit(X, y)
```

## popular machine learning libraries
these libraries allow you to practically import your data once and run
it on any number of both popular and obscure ML algorithms; they also
have adapters for calling from many popular languages, including Ruby
- [Weka](http://www.cs.waikato.ac.nz/ml/weka/)
- [Shogun](https://github.com/shogun-toolbox/shogun) is on Github but looks like it's a giant pain to use, with
  confusing documentation
- [scikit-learn](http://scikit-learn.org/)
- [Matlab](http://www.mathworks.com/products/matlab/)? but not open source, obviously

## what this means to us
- rescue blob spikes
^ we could use a particular kind of SVM called a one-class SVM, which is
a bit different from a two-class SVM in that you don't need to label the
training data yourself. we could use it in order to watch out for when
we do a production deploy and the numbers of rescue blobs all of a
sudden go through the roof. this is called anomaly detection or novelty
detection. see [scikit-learn](http://scikit-learn.org/stable/modules/svm.html#density-estimation-novelty-detection) for more info.

## summary
1. we went over a demo
2. we reviewed fundamental concepts of machine learning
3. how SVMs work
4. how to train SVMs
5. we went over some classic examples of where SVMs are currently used
6. we went over some publically available SVM implementations

## i have books available if anyone wants to borrow them

## thanks go to gessner, data will, and josh

## Q & A
