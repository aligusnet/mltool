import Test.Framework (defaultMain, testGroup)


import qualified MachineLearning.RegressionTest as Regression
import qualified MachineLearning.Regression.LeastSquaresTest as LeastSquares
import qualified MachineLearning.Regression.LogisticTest as Logistic
import qualified MachineLearning.Regression.GradientDescentTest as GradientDescent
import qualified MachineLearning.NeuralNetworkTest as NeuralNetwork
import qualified MachineLearning.PCATest as PCA

main = defaultMain tests

tests = [
  testGroup "MachineLearning.Regression" Regression.tests
  , testGroup "MachineLearning.Regression.LeastSquares" LeastSquares.tests
  , testGroup "MachineLearning.Regression.Logistic" Logistic.tests
  , testGroup "MachineLearning.Regression.GradientDescent" GradientDescent.tests
  , testGroup "MachineLearning.NeuralNetwork" NeuralNetwork.tests
  , testGroup "MachineLearning.PCA" PCA.tests
  ]
