# SVMs
## gabe heafitz
^ very brief mention of what SVMs are: a popular, easy-to-use machine learning algorithm

---

## plan of attack
1. demo of SVM running on some example data
1. interpreting the results
1. review of fundamental concepts of machine learning; how is machine learning helpful?
1. how do SVMs work?
1. how do you train an SVM?
1. some classic usage examples
1. some publically available implementations

---

# demo
## train an SVM against example data

---

# well, first, what does our data look like?
## heart dataset from UCI
```
63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0
67.0,1.0,4.0,160.0,286.0,0.0,2.0,108.0,1.0,1.5,2.0,3.0,3.0,2
67.0,1.0,4.0,120.0,229.0,0.0,2.0,129.0,1.0,2.6,2.0,2.0,7.0,1
37.0,1.0,3.0,130.0,250.0,0.0,0.0,187.0,0.0,3.5,3.0,0.0,3.0,0
41.0,0.0,2.0,130.0,204.0,0.0,2.0,172.0,0.0,1.4,1.0,0.0,3.0,0
56.0,1.0,2.0,120.0,236.0,0.0,0.0,178.0,0.0,0.8,1.0,0.0,3.0,0
```

^ this is the first 6 rows of data from a dataset from UCI on medical patients who were examined for whether they have heart disease. and this actually isn't even all of the columns in the original dataset -- there were some 70-odd columns there; this is just the most relevant 13 or so columns of data that most statistics or machine learning studies to use this dataset have actually used.

---

# what does each column represent?

```
age  sex     bp                  hr                        ? 
63.0 1.0 1.0 145.0 233.0 1.0 2.0 150.0 0.0 2.3 3.0 0.0 6.0 1
67.0 1.0 4.0 160.0 286.0 0.0 2.0 108.0 1.0 1.5 2.0 3.0 3.0 0
67.0 1.0 4.0 120.0 229.0 0.0 2.0 129.0 1.0 2.6 2.0 2.0 7.0 1
37.0 1.0 3.0 130.0 250.0 0.0 0.0 187.0 0.0 3.5 3.0 0.0 3.0 0
41.0 0.0 2.0 130.0 204.0 0.0 2.0 172.0 0.0 1.4 1.0 0.0 3.0 0
56.0 1.0 2.0 120.0 236.0 0.0 0.0 178.0 0.0 0.8 1.0 0.0 3.0 0
```

^ 1. age 2. sex 3. chest pain type (4 values) 4. resting blood pressure 5. serum cholesterol in mg/dl 6. fasting blood sugar > 120 mg/dl 7. resting electrocardiographic results (values 0,1,2) 8. maximum heart rate achieved 9. exercise induced angina 10. ST depression induced by exercise relative to rest; 11. the slope of the peak exercise ST segment, 12. number of major vessels (0-3) colored by flourosopy; 13. thal: 3 = normal; 6 = fixed defect; 7 = reversable defect; 13. 2 if heart disease, 1 if no heart disease.

---

1. age
2. sex
3. chest pain type  (4 values)
4. resting blood pressure
5. serum cholesterol in mg/dl
6. fasting blood sugar > 120 mg/dl
7. resting electrocardiographic results  (values 0,1,2)
8. maximum heart rate achieved
9. exercise induced angina
10. ST depression induced by exercise relative to rest
11. the slope of the peak exercise ST segment
12. number of major vessels (0-3) colored by flourosopy
13.  thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

---

# okay, finally ready to SVM

---

![115%](heart_demo.mov)

---

# demo
## evaluate the results
^ the absolute baseline performance we could expect from the algorithm is if we just predicted the same for every example in the validation set the most common response from the training set. so, in this case, 76 out of the 170 examples in the training set correspond to cases in which the patient did in fact have heart disease, and 94 out of the 170 examples in the training set correspond to patients without heart disease. as a result, the most common case in the training data was that the patient did not have heart disease. you could therefore just predict there is no heart disease for all examples, since that is what is most common. and you'd be right 56% of the time for the validation set that we're using. (the validation set is the subset of data that we've held out of training the learner. it's supposed to give us a good idea of how our algorithm would perform in production, against *truly* unknown data.) but there's not much nuance or logic in just predicting whatever was the most common. we can do better than that (and if your learner isn't even beating baseline, something is wrong). in this case, our basic SVM got 85% of the predictions right for the validation set. not bad, for an off-the-shelf library we just shoved a bunch of data into, compiled in less than a second, trained in less than a second, and got predictions from in less than a second. But best of all, it saved us the trouble of researching and explicitly coding findings from modern medicine instead.

---

# what ML is for
## finding patterns in the data
^ machine learning algorithms earn their keep by finding patterns in raw data (well, raw data that has been encoded in a way that the machine learning algorithm can ingest, and also, in many cases, has also been massaged to remove data deemed irrelevant or to normalize data with high variance)

---

machine learning uses statistics. in engineering, we're used to solving a deterministic problem where our solution solves the problem all the time. if we're asked to write software to control a vending machine, it had better work all the time, regardless of the money entered or the buttons pressed. there are many problems where the solution isn't deterministic.

---

that is, we don't know enough about the problem or don't have enough computing power to properly model the problem. for these problems we need statistics.
-- from Machine Learning in Action, by Peter Harrington

---

# terminology
![](software_terminology.jpg)

---

# attribute
## one of the columns in your data
examples from the heart data set:
- age
- gender
- chest pain type (this is a set of 4 nominal values)
- resting blood pressure
- serum cholesterol in mg/dL
- fasting blood sugar > 120 mg/dL

---

# classification
## tell me which of 2 classes the row of data fits in
example from the heart data set:
whether heart disease is present (true) or absent (false)

^ note that we're just talking about 2-class classification here. there are lots of cases in which you want to predict which of n classes a particular row of data fits in, but today, we'll just focus on 2-class, or binary, classification.

---

# 2 attributes of the heart data set

- age
- maximum heart rate

^ we'll start by discussing just these 2 arbitrarily chosen attributes, but later, we're going to discuss training models that take potentially many more than 2 attributes in a data set into account.

---

![inline](heart_rate_by_age.png)

---

# supervised vs. unsupervised
## whether your data is already labeled with the "answers"
^ in supervised learning, you are training an algorithm on a bunch of data to which you already have the "answers" in unsupervised learning, such as clustering, you are asking a learner to find patterns and commonalities without your inputting the labels (or "answers") yourself but today, we're just going to focus on supervised learning

---

# data is divided into subsets
1. training
1. validation
1. holdout set
1. production data

^ the data you have on hand when training your model is divided into these four subsets, but this isn't a strict rule. naturally, as you can imagine, they all have a part in the ML process, and you proceed in order from top to bottom.

---

# training data
^ this is a subset of the data you have at the time of training the machine learning algorithm. again, since we're dealing with supervised learning, this data is already labeled as to whether each row represents someone with heart disease or not, for example. in general, the more training data you have, the better, since it allows you to train a more generalized model and one that is less prone to variance in a couple of data points. that is, if you only had a small sample of data on which to train your model, your model might weight a little variation in one of the attributes too sensitively or not sensitively enough, for example.

---

# validation data

^ this is a subset of the data you do not use for training the model.  instead, you run the trained model against it, and because it wasn't used in training the model, the idea is that it will give you a decent idea of how your model would perform if you ever used it in production on unknown (meaning, unlabeled) data. however, because you train models using different parameters or configurations and then run those models on the validation data, and you repeat this process several times until you've found a set of parameters or a configuration that performs well, you can no longer say that the validation data was not involved in the training of the model.

---

# holdout set

^ that's why you set aside other data for later, to get a much more unbiased evaluation of how your model performs.  i would like to note that the terms "validation set" and "holdout set", or "validation data" and "holdout data" are often used interchangeably. i'll try to stick with "validation" here, but if i say "holdout" instead, just keep in mind that the two serve largely the same purpose.

---

# overfitting[^3]
![inline](plot_underfitting_overfitting_001.png)

^ a goldilocks and the three bears kind of situation applies to machine learning. has your model not approximated the true data closely enough (by not giving the signal enough weight), has it approximated it _too_ well (by giving noise too much weight), or has it approximated it just right? not that one is necessarily worse than the other, but we do especially watch out for overfitting in training our models and employ tactics to avoid it. i think it's because we'd prefer to be pessimists and say we don't know than be optimists and make claims our models can't necessarily support.

[^3]: scikit learn. (2015). Underfitting vs. Overfitting. Retrieved April 7th, 2015, from [scikit-learn.org](http://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html).

---

# knowledge representation
can a human interpret the patterns that the machine has found, or is the model an opaque black box?

---

## how to train and use a machine learning model[^1]
[^1]: Harrington, M. (2012). Machine Learning In Action. Shelter Island, NY: Manning Publications Co.

---

# step 1
## collect a data set

---

# step 2
## encode the data set in such a way that the machine learning algorithm will be able to ingest

---

# step 3
## filter out any known noise or other useless attributes or rows in the data, so that the machine learning algorithm doesn't get distracted by something you know to be irrelevant

---

# step 4
## train the algorithm: "you feed the algorithm good clean data from the first two steps and extract knowledge or information." the "knowledge" or "information" *is* your model

---

# step 5
## exercise your model on the validation set -- this will show how well your model performs with unknown data

^ has your model overfit the training data, or is it a model that reflects a truly general pattern in the data and will perform well in the future when supplied with data points that have some variance in their attribute values?

---

# step 6
## use your model in production. you may want to periodically re-train your model, as you collect more data in your system

---

# step 7[^2]
## profit?

[^2]: This one's not from the Harrington book. I added this one myself.

---

# how SVMs work

---

# linear kernel[^4]
![](plot_separating_hyperplane_001.png)

---

![inline](plot_separating_hyperplane_001.png)

^ generally, a linear kernel requires the least amount of time to train, and works best if your data is linearly separable. it works very much like trying to position an infinitely long yardstick in such a way that it will be the farthest away from all of the datapoints; the datapoints closest to it that most influence its position and slope are the support vectors and are the reason we call this learning algorithm a support vector machine.  however, the metaphor of the infinitely long yardstick can only be applied literally if you have 2 attributes in your data, because you're trying to separate two 2-dimensional clouds of data points. but the heart data we were looking at earlier had not 2 attributes, but 13!! how do we get there? well, if you have 2 attributes, or columns of data, you're positioning an infinite yardstick. if you have 3 attributes, or columns of data, you're trying to position an infinite plane. if you have 4 attributes, you're trying to position a plane in 4 dimensions (not that we can easily visualize that), and so on. so, as far as the heart data goes, you're trying to position a flat plane in 13-dimensional space. but whether you're talking about a line or a plane, regardless of the number of dimensions, we call the line or plane that separates the data the separating hyperplane. but it's the separating hyperplane that we're trying to find throughout all this, so that in the future, we can plot an unclassified or uncategorized data point, and, based on which side of the hyperplane the data point falls, we will have a prediction for which class or category applies to it.

[^4]: scikit learn. (2015). SVM: Maximum margin separating hyperplane. Retrieved April 7th, 2015, from [scikit-learn.org](http://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html#example-svm-plot-separating-hyperplane-py).

---

# math[^5]

$$
\left\lbrace
\begin{array}{l l}
\min_{\vec{w},b,\vec{\xi}} & \frac{1}{2} ||\vec{w}||^2+ C\sum \xi_i \\
\forall i \in T, & y_i(K(\vec{w},\vec{x_i}) + b) \geq 1 - \xi_i\\
\forall i \in T, & \xi_i \geq 0
\end{array}
\right.
$$

[^5]: thefiletree.com. (2015). An application of Sequential Minimum Optimization on mail classification. Retrieved April 8th, 2015, from [thefiletree.com](https://thefiletree.com/Francois/SVM.tex).

---

# demo of linear kernel SVM

![150%](svm_toy_linear_kernel_demo_0.mov)

---

# but what if the data is not linearly separable?
## slack variables allow us some room for error and to find a line that doesn't necessarily separate the data perfectly, but still to do the best job we can

---

# slack allows for some error

![150%](svm_toy_linear_kernel_demo_1.mov)

---

# what if the data are _really_ not linearly separable?

![150%](svm_toy_linear_kernel_demo_2.mov)

^ we've looked at trying to fit a line or a plane to the data in such a way that will best separate one class (say, no heart disease) from the other class (presence of heart disease). we've also discussed what happens if the data overlap a bit and you have to fit the line or plane in such a way that a couple of the training data points fall on the wrong side, with slack variables. but there are cases in which even with slack variables, you still will end up with pretty bad performance. head back to [the LibSVM page](http://www.csie.ntu.edu.tw/~cjlin/libsvm/index.html?js=1#svm-toy-js); make sure to set t = 0 to get a linear kernel; draw a parabolic cloud of data and another cloud of data that surrounds it.

---

# what if the data are _really_ not linearly separable?
## higher order SVMs
- polynomial function
- Gaussian radial basis function

^ these higher order kernels take longer to train, but they are much more powerful than linear kernels for fitting more complex datasets.  depending on your data, you can get much more accurate predictions on your unknown data.

---

# higher order SVMs
![150%](svm_toy_higher_dimension_kernels_demo_0.mov)

---

# what is going on here?

^ how do these higher order kernels, the polynomial and Gaussian functions work? 

---

# data that can't possibly be separated by a straight line[^6]
![inline](linear_non_separable.png)

[^6]: Mueller, A. (2012, December 26). Kernel Approximations for Efficient SVMs (and other feature extraction methods) [update]. Message posted to http://http://peekaboo-vision.blogspot.com/

^ here you have three data points in 2 dimensions that can't possibly be separated by a straight line. but what if you BENT the 2 dimensional space?

---

# data that can't possibly be separated by a straight line[^6]
![inline](linear_non_separable_projected_2D.png)

^ bang. if you apply one of these higher order kernel functions to your data, you can find a straight line, or hyperplane, that separates your data.

---

## another example, this time in 3-D
![140%](svm_toy_higher_dimension_kernels_1.mov)

---

![](spoon-boy.jpg)

---

![](keanu_whoa.jpg)

---

# gaussian radial basis function parameters
- C: the cost parameter
- gamma: controls sharpness of bumps[^7]

^ we try to home in on the best performing combinations of C (the cost parameter, or how willing we are to settle for a margin that doesn't perfectly separate the data points in the training set) and gamma. "A small gamma gives you a pointed bump in the higher dimensions, a large gamma gives you a softer, broader bump. So a small gamma will give you low bias and high variance while a large gamma will give you higher bias and low variance." we typically search a grid of possible combinations of C and gamma in order to find the best performing values.

[^7]: Quora. (2015). What are C and gamma with regards to a support vector machine?. Retrieved April 7th, 2015, from [quora.com](https://www.quora.com/What-are-C-and-gamma-with-regards-to-a-support-vector-machine?share=1).

---

## C, the cost parameter
![150%](svm_toy_gaussian_cost_parameter_demo.mov)

---

## gamma, the spikiness parameter
![150%](svm_toy_gaussian_gamma_parameter_demo.mov)

---

# SVMs: the advantages and disadvantages

---

# advantages
- widely considered to be the best "stock" learning algorithm[^1][^8]
- relatively very simple to configure and require very little tuning of parameters

^ you can take the SVM algorithm right off the shelf, and train a learner on your data and get very good or excellent performance on it. witness the performance of SVMs on analyzing handwritten digits: an SVM with a relatively simple configuration correctly identified a handwritten digit ~98.6% of the time, whereas a top-of-the-line artificial neural network delivered ~99.75% accuracy. the artificial neural network therefore made 1/6 the number of errors that the SVM did, but still, that ~98.6% figure sure isn't bad for an off-the-shelf algorithm.

[^8]: lecun.com. (2015). The MNIST Database of handwritten digits. Retrieved April 8th, 2015, from [lecun.com](http://yann.lecun.com/exdb/mnist/).

---

![](machine_comparison.jpg)

---

![](machine_comparison_2.jpg)

---

# disadvantages

- black box: tough to extract knowledge representation[^1]

^ as opposed to some other machine learning algorithms, such as decision trees, the various weights an SVM assigns to the attributes in your data are not something you can take away an understanding from, draw conclusions from, and take action on. you can't extract knowledge on, say, which attributes of the heart dataset are most highly correlated with risk of heart disease so that you can go issue public health advisories, or which attributes of your sales data highlight the best sales leads so that you can advise your marketing team accordingly. so, it depends on whether you just want an answer (classify an unknown data point into a certain category) or, instead, you want greater understanding from your data.

---

# disadvantages
- can be slow[^10]

[^10]: scikit-learn. (2015). sklearn.svm.SVC. Retrieved April 8th, 2015, from [scikit-learn.org](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).

---

# disadvantages
- but not too bad if you're using a linear kernel[^11]

[^11]: Quora. (2015). Why is kernelized SVM much slower than linear SVM?. Retrieved April 7th, 2015, from [quora.com](https://www.quora.com/Why-is-kernelized-SVM-much-slower-than-linear-SVM).

^ according to [scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html), the time to train a higher dimension SVM grows by more than O(n^2) (where n is the number of training examples), so, much beyond 10,000 examples, you're looking at a long time to train. however, with a linear SVM (one where you're just trying to linearly separate data, like with a yardstick), the performance is O(n).

---

# popular real-world applications of SVMs[^12]

- text categorization
- image classification
- protein classification in medical research
- handwritten character recognition

[^12]: Wikipedia. (2015). Support Vector Machine. Retrieved April 7th, 2015, from [wikipedia.org](https://en.wikipedia.org/wiki/Support_vector_machine).

---

# popular SVM implementations
- [LibSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/index.html)
- [SVMLight](http://svmlight.joachims.org/)
- [scikit-learn](http://scikit-learn.org/)

---

## not very difficult to use in your own code
- these libraries have somewhat clear APIs
- python example, using scikit-learn[^13]

```python
from sklearn.svm import SVC

X = ... # my training data
y = ... # my training labels (the class or classification of each training data point)

clf = SVC(C=1.0)
clf.fit(X, y)

X_test = ... # my production data

predictions = clf.predict(X_test)

# done!
```

[^13]: [data will](mailto:wwolf@shopkeep.com)

---

# popular machine learning libraries
## import your data once and run it on any number of both popular and obscure ML algorithms
## adapters for calling from many popular languages, including Ruby

- [Weka](http://www.cs.waikato.ac.nz/ml/weka/) -- written by kiwis
- [Shogun](https://github.com/shogun-toolbox/shogun) -- on github but looks like it's a giant pain to use, with confusing documentation
- [scikit-learn](http://scikit-learn.org/)
- [Matlab](http://www.mathworks.com/products/matlab/) -- but not open source

---

# okay, so, wonderful, but what can _we_ do with it?
- rescue blob spikes
- handwriting recognition
- e-commerce fraud detection?
- identifying the most promising sales leads
- identifying whether a PR comment is random text or avrohom attempting to communicate

^ we could use a particular kind of SVM called a one-class SVM, which is a bit different from a two-class SVM in that you don't need to label the training data yourself. we could use it in order to watch out for when we do a production deploy and the numbers of rescue blobs all of a sudden go through the roof. this is a kind of unsupervised learning called anomaly detection or novelty detection. see [scikit-learn](http://scikit-learn.org/stable/modules/svm.html#density-estimation-novelty-detection) for more info.

---

# summary cum laude
1. demo of SVM running on some example data
1. interpreting the results
1. review of fundamental concepts of machine learning
1. how do SVMs work?
1. how do you train an SVM?
1. some classic usage examples
1. some publically available implementations

---

# i have books available to lend/give you

![inline fit](machine_learning_in_action.jpg)![inline fit](machine_learning_tom_mitchell.jpg)

---

# many thanks go to data will, gessner, and josh

---

# question time!
## examples:
- are SVMs a good source of riboflavin?
- can i wash my cat with an SVM?

---
# fin

![original](Dorsal_fin_01.jpg)
