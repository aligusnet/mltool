## Machine Learning Toolbox

### Build documentation

    stack haddock

### Build the project

    stack build

### Run samples app

Please run sample app from root dir (because paths to training data sets are hardcoded).

    stack exec linreg  # Linear Regression Sample App
    stack exec logreg  # Logistic Regression (Classification) Sample App
    stack exec digits  # Muticlass Classification Sample App
                       # (Recognition if Handwritten Digits)

### Run unit tests

    stack test
