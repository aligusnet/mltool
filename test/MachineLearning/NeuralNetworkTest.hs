module MachineLearning.NeuralNetworkTest
(
  tests
)

where

import Test.Framework (testGroup)
import Test.Framework.Providers.HUnit
import Test.HUnit
import Test.HUnit.Approx
import Test.HUnit.Plus

import MachineLearning.DataSets (dataset2)

import qualified Control.Monad.Random as RndM
import qualified Numeric.LinearAlgebra as LA
import qualified MachineLearning as ML
import qualified MachineLearning.Optimization as Opt
import MachineLearning.Model
import MachineLearning.NeuralNetwork
import qualified MachineLearning.NeuralNetwork.Sigmoid as Sigmoid

(x, y) = ML.splitToXY dataset2

nnt = Sigmoid.makeTopology (LA.cols x) 2 [10]
model = NeuralNetwork nnt

gradientCheckingEps = 0.1

checkGradientTest lambda = do
  let thetas = Sigmoid.initializeTheta 1511197 nnt
      diffs = take 5 $ map (\e -> Opt.checkGradient model lambda x y thetas e) [0.005, 0.0051 ..]
      diff = minimum $ filter (not . isNaN) diffs
  assertApproxEqual (show thetas) gradientCheckingEps 0 diff


flattenTest = do
  theta <- Sigmoid.initializeThetaIO nnt
  let theta' = flatten $ unflatten nnt theta
      norm = LA.norm_2 (theta - theta')
  assertApproxEqual "flatten" 1e-10 0 norm


xPredict = LA.matrix 2 [ -0.5, 0.5
                       , 0.2, -0.2
                       , 1, 1
                       , 1, 0
                       , 0, 0]
yExpected = LA.vector [1, 1, 0, 0, 1]

learnTest minMethod =
  let lambda = 0.5 / (fromIntegral $ LA.rows x)
      x1 = ML.mapFeatures 2 x
      nnt = Sigmoid.makeTopology (LA.cols x1) 2 [10]
      model = NeuralNetwork nnt
      xPredict1 = ML.mapFeatures 2 xPredict
      initTheta = Sigmoid.initializeTheta 5191711 nnt
      (theta, optPath) = Opt.minimize minMethod model 1e-7 100 lambda x1 y initTheta
      yPredicted = hypothesis model xPredict1 theta
      js = (LA.toColumns optPath) !! 1
  in do
    assertVector (show js) 0.01 yExpected yPredicted


tests = [ testGroup "gradient checking" [
            testCase "non-zero lambda" $ checkGradientTest 0.01
            , testCase "zero lambda" $ checkGradientTest 0
              ]
        , testGroup "flatten" [
            testCase "flatten" flattenTest
            ]
        , testGroup "learn" [
            testCase "BFGS" $ learnTest (Opt.BFGS2 0.01 0.7)
            ]
        ]
