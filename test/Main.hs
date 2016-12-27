import Test.Framework (defaultMain, testGroup)


import qualified MachineLearning.RegressionTest as Regression
import qualified MachineLearning.Regression.LeastSquaresTest as LeastSquares
import qualified MachineLearning.Regression.LogisticTest as Logistic

main = defaultMain tests

tests = [
  testGroup "MachineLearning.Regression" Regression.tests
  , testGroup "MachineLearning.Regression.LeastSquares" LeastSquares.tests
  , testGroup "MachineLearning.Regression.Logistic" Logistic.tests
  ]
