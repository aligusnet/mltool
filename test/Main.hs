import Test.Framework (defaultMain, testGroup)


import qualified MachineLearningTest as MachineLearning
import qualified MachineLearning.RegressionTest as Regression
import qualified MachineLearning.Classification.BinaryTest as Classification.Binary
import qualified MachineLearning.Classification.OneVsAllTest as Classification.OneVsAll
import qualified MachineLearning.LeastSquaresModelTest as LeastSquaresModel
import qualified MachineLearning.LogisticModelTest as LogisticModel
import qualified MachineLearning.MultiSvmModelTest as MultiSvmModel
import qualified MachineLearning.Optimization.GradientDescentTest as GradientDescent
import qualified MachineLearning.NeuralNetworkTest as NeuralNetwork
import qualified MachineLearning.PCATest as PCA
import qualified MachineLearning.ClusteringTest as Clustering
import qualified MachineLearning.RandomTest as Random

main = defaultMain tests

tests = [
  testGroup "MachineLearning" MachineLearning.tests
  , testGroup "MachineLearning.Regression" Regression.tests
  , testGroup "MachineLearning.Classification.Binary" Classification.Binary.tests
  , testGroup "MachineLearning.Classification.OneVsAll" Classification.OneVsAll.tests
  , testGroup "MachineLearning.LeastSquaresModel" LeastSquaresModel.tests
  , testGroup "MachineLearning.LogisticModel" LogisticModel.tests
  , testGroup "MachineLearning.MultiSvmModel" MultiSvmModel.tests
  , testGroup "MachineLearning.Optimization.GradientDescent" GradientDescent.tests
  , testGroup "MachineLearning.NeuralNetwork" NeuralNetwork.tests
  , testGroup "MachineLearning.PCA" PCA.tests
  , testGroup "MachineLearning.Clustering" Clustering.tests
  , testGroup "MachineLearning.Random" Random.tests
  ]
