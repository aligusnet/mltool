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
import qualified MachineLearning.NeuralNetwork.TopologyMaker as TM

(x, y) = ML.splitToXY dataset2

gradientCheckingEps = 0.1

checkGradientTest eps activation loss lambda = do
  let nnt = TM.makeTopology activation loss (LA.cols x) 2 [10]
      model = NeuralNetwork nnt
      thetas = initializeTheta 1511197 nnt
      diffs = take 5 $ map (\e -> Opt.checkGradient model lambda x y thetas e) [0.005, 0.0051 ..]
      diff = minimum $ filter (not . isNaN) diffs
  assertApproxEqual (show thetas) eps 0 diff


xPredict = LA.matrix 2 [ -0.5, 0.5
                       , 0.2, -0.2
                       , 1, 1
                       , 1, 0
                       , 0, 0]
yExpected = LA.vector [1, 1, 0, 0, 1]

learnTest activation loss minMethod nIters =
  let lambda = 0.5 / (fromIntegral $ LA.rows x)
      x1 = ML.mapFeatures 2 x
      nnt = TM.makeTopology activation loss (LA.cols x1) 2 [10]
      model = NeuralNetwork nnt
      xPredict1 = ML.mapFeatures 2 xPredict
      initTheta = initializeTheta 5191711 nnt
      (theta, optPath) = Opt.minimize minMethod model 1e-7 nIters lambda x1 y initTheta
      yPredicted = hypothesis model xPredict1 theta
      js = (LA.toColumns optPath) !! 1
  in do
    assertVector (show js) 0.01 yExpected yPredicted


tests = [ testGroup "gradient checking" [
            testCase "Sigmoid: non-zero lambda" $ checkGradientTest 0.1 TM.ASigmoid TM.LSigmoid 0.01
            , testCase "Sigmoid: zero lambda" $ checkGradientTest 0.1 TM.ASigmoid TM.LSigmoid 0
            , testCase "ReLU - Softmax: non-zero lambda" $ checkGradientTest 12 TM.ARelu TM.LSoftmax 0.01
            , testCase "ReLU - Softmax: zero lambda" $ checkGradientTest 12 TM.ARelu TM.LSoftmax 0
              ]
        , testGroup "learn" [
            testCase "Sigmoid: BFGS" $ learnTest TM.ASigmoid TM.LSigmoid (Opt.BFGS2 0.01 0.7) 50
            , testCase "ReLU - Softmax: BFGS" $ learnTest TM.ARelu TM.LSoftmax (Opt.BFGS2 0.1 0.1) 50
            , testCase "Tanh - Softmax: BFGS" $ learnTest TM.ATanh TM.LSoftmax (Opt.BFGS2 0.1 0.1) 50
            ]
        ]
