import Test.Framework (defaultMain, testGroup)


import qualified MachineLearningTest as MachineLearning
import qualified MachineLearning.RegressionTest as Regression
import qualified MachineLearning.ClassificationTest as Classification
import qualified MachineLearning.LeastSquaresModelTest as LeastSquaresModel
import qualified MachineLearning.LogisticModelTest as LogisticModel
import qualified MachineLearning.Optimization.GradientDescentTest as GradientDescent
import qualified MachineLearning.NeuralNetworkTest as NeuralNetwork
import qualified MachineLearning.PCATest as PCA
import qualified MachineLearning.ClusteringTest as Clustering
import qualified MachineLearning.RandomTest as Random

main = defaultMain tests

tests = [
  testGroup "MachineLearning" MachineLearning.tests
  , testGroup "MachineLearning.Regression" Regression.tests
  , testGroup "MachineLearning.Classification" Classification.tests
  , testGroup "MachineLearning.LeastSquaresModel" LeastSquaresModel.tests
  , testGroup "MachineLearning.LogisticModel" LogisticModel.tests
  , testGroup "MachineLearning.Optimization.GradientDescent" GradientDescent.tests
  , testGroup "MachineLearning.NeuralNetwork" NeuralNetwork.tests
  , testGroup "MachineLearning.PCA" PCA.tests
  , testGroup "MachineLearning.Clustering" Clustering.tests
  , testGroup "MachineLearning.Random" Random.tests
  ]
