---
title: "Final Project: Image Classification"
author: "Taylor Arnold"
---

**Due**: 2018-04-24 (5pm)

The goal of this project is to start with a corpus of images, build a
classification algorithm for the corpus, and evaluate how well it works.
You may either construct a smaller dataset manually (a mix of photos found
online or taken directly by you) or to start with a preconstructed dataset.
Details on the data options are given below. You should upload your script and
completed HTML file to GitHub ahead of class. We will present everyone's
results in class on the 25th.

You may work in pairs. You will write the same report, but please upload the
document to each of your separate repositories to make grading easier.

**Datasets** The more straight-forward task is to select a dataset that has
already been compiled for you. Below are the options that I suggest you
select from. If you want to use something else you must before getting started.
You can find the data and more information about all of them on
the fast.ai website:

- [https://course.fast.ai/datasets](https://course.fast.ai/datasets)

The minimum number of categories shows the number of categories
and particular subsets you should include in the dataset.

- Oxford-IIIT Pet (37 categories; about 200 images each; use the whole dataset)
- Imagewoof (10 categories; use 320 pixel size; use all of the data and the predefined train/validation splits)
- Oxford 102 Flowers (102 categories; 40-258 images each; use at least 20 categories)
- Food-101 (101 food categories; 750 training images each; use at least 10 categories)
- Caltech 101 (101 categories; around 50 images each; use at least )
- Caltech-UCSD Birds (200 bird species; about 60 images each; use at least 30 categories)
- Stanford cars (196 categories; around 40 training images each; use at least 20 categories)

If you choose to construct your own dataset, you should first select at least
3 (but ideally 5 or more) classes that you want to categorize. Then, take or
find at least 75 different photos from each class (with a minimum of 200 photos
overall; 300 for pairs). You must talk to me about your idea before
collecting data.

In either case, you should manually construct a training portion of the data
and a validation portion of the data (expect for Imagewoof; use their pre-defined
splits). I suggest something between at 50/50 and 80/20 split (lean towards the
80/20 for data you construct and towards 60/40 on data from others).

**Reports** You are going to write your report as an Rmarkdown file, with
embedded graphics and code within the document. The style of the report should
follow that of a machine learning tutorial. Here are some examples (these are
longer and a bit more technical that yours needs to be, but you will see the
general tone and style):

- [https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
- [https://towardsdatascience.com/image-classification-python-keras-tutorial-kaggle-challenge-45a6332a58b8](https://towardsdatascience.com/image-classification-python-keras-tutorial-kaggle-challenge-45a6332a58b8)

Note that these are written in full sentences, with code and output embedded
throughout the piece in a meaningful way. In terms of the specific content in
the report you should include the following:

- what the dataset is that you are using and some example images from the
various classes (if you have a lot of classes, you can just show a subset)
- the results using your own neural network trained from scratch; classification
rates as well as confusion matrices (where feasible)
- results using transfer learning; perhaps from multiple layers, multiple
original models, or multiple ways of using the embedded weights
- images with the highest probabilities; selected images that were
misclassified from the validation set
- visualization of the embedding or final model using t-SNE or PCA analysis.

If some of the steps take a long time to run, consider saving the results and
calling pre-trained versions from within the code. I will illustrate this in
class on the 18th.
