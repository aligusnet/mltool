## Machine Learning Toolbox

[![Build Status](https://travis-ci.org/Alexander-Ignatyev/mltool.svg?branch=master)](https://travis-ci.org/Alexander-Ignatyev/mltool)
[![Coverage Status](https://coveralls.io/repos/github/Alexander-Ignatyev/mltool/badge.svg)](https://coveralls.io/github/Alexander-Ignatyev/mltool)
[![Documentation](https://img.shields.io/badge/mltool-documentation-blue.svg)](https://alexander-ignatyev.github.io/mltool-docs/doc/index.html)
[![Hackage](https://img.shields.io/hackage/v/mltool.svg)](https://hackage.haskell.org/package/mltool)

### Supported Methods and Problems

#### Supervised Learning

##### Regression Problem

* Normal Equation;

* Linear Regression using Least Squares approach.

##### Classification Problem

* Softmax Classifier;

* Multi SVM Classifier;

* Logistic Regression;

* Neural Networks, please see the details below.

#### Unsupervised Learning

* Principal Component Analysis (Dimensionality reduction problem);

* K-Means (Clustering).

#### Neural Networks

* Activations: ReLu, Tanh, Sigmoid;

* Loss Functions: Softmax, Multi SVM, Logistic.

### Usage

#### OS X/macOS prerequisites setup

* Using [Homebrew](https://brew.sh/):

```
brew install pkg-config gsl
```

or 

* Using [MacPorts](https://www.macports.org/):

```
sudo port install pkgconfig gsl
```

#### Build the project

    stack build

#### Run examples app

Please run sample app from root dir (because paths to training data sets are hardcoded).

```bash
cd examples
stack build
stack exec linreg      # Linear Regression Sample App
stack exec logreg      # Logistic Regression (Classification) Sample App
stack exec digits      # Muticlass Classification Sample App
                       # (Recognition of Handwritten Digitts
stack exec digits-pca  # Apply PCA dimensionaly reduction to digits sample app
stack exec digits-svm  # Support Vector Machines
stack exec nn          # Neural Network Sample App
                       # (Recognition of Handwritten Digits)
stack exec kmeans      # Clustering Sample App
```

#### Run unit tests

    stack test


### Examples

* Linear Regression: [source code](https://github.com/Alexander-Ignatyev/mltool/blob/master/examples/linear_regression/Main.hs);

* Logistic Regression: [source code](https://github.com/Alexander-Ignatyev/mltool/blob/master/examples/logistic_regression/Main.hs);

* Multiclass Logistic Regression: [source code](https://github.com/Alexander-Ignatyev/mltool/blob/master/examples/digits_classification/Main.hs);

* Multiclass Logistic Regression with PCA: [source code](https://github.com/Alexander-Ignatyev/mltool/blob/master/examples/digits_classification_pca/Main.hs);

* Multiclass Support Vector Machine: [source code](https://github.com/Alexander-Ignatyev/mltool/blob/master/examples/digits_classification_svm/Main.hs);

* Neural Networks: [source code](https://github.com/Alexander-Ignatyev/mltool/blob/master/examples/neural_networks/Main.hs);

* K-Means: [source code](https://github.com/Alexander-Ignatyev/mltool/blob/master/examples/kmeans/Main.hs).
