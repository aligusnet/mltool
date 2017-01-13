import Test.Framework (defaultMain, testGroup)


import qualified MachineLearningTest as MachineLearning
import qualified MachineLearning.RegressionTest as Regression
import qualified MachineLearning.Regression.LeastSquaresTest as LeastSquares
import qualified MachineLearning.Regression.LogisticTest as Logistic
import qualified MachineLearning.Regression.GradientDescentTest as GradientDescent
import qualified MachineLearning.NeuralNetworkTest as NeuralNetwork
import qualified MachineLearning.PCATest as PCA
import qualified MachineLearning.ClusteringTest as Clustering
import qualified MachineLearning.RandomTest as Random

main = defaultMain tests

tests = [
  testGroup "MachineLearning" MachineLearning.tests
  , testGroup "MachineLearning.Regression" Regression.tests
  , testGroup "MachineLearning.Regression.LeastSquares" LeastSquares.tests
  , testGroup "MachineLearning.Regression.Logistic" Logistic.tests
  , testGroup "MachineLearning.Regression.GradientDescent" GradientDescent.tests
  , testGroup "MachineLearning.NeuralNetwork" NeuralNetwork.tests
  , testGroup "MachineLearning.PCA" PCA.tests
  , testGroup "MachineLearning.Clustering" Clustering.tests
  , testGroup "MachineLearning.Random" Random.tests
  ]
