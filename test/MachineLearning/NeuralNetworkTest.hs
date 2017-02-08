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

(x, y) = ML.splitToXY dataset2

x1 = ML.addBiasDimension x
nnt = makeTopology (LA.cols x) 2 [10]
model = NeuralNetwork nnt

gradientCheckingEps = 0.1


thetaSizeTest = do
  thetas <- RndM.evalRandIO $ initializeThetaListM nnt
  let sizesActual = map LA.size thetas
      sizesExpected = getThetaSizes nnt
  assertEqual "theta sizes" sizesExpected sizesActual


checkGradientTest lambda = do
  let thetas = initializeTheta 1511197 nnt
      diffs = take 5 $ map (\e -> Opt.checkGradient model lambda x1 y thetas e) [0.005, 0.0051 ..]
      diff = minimum $ filter (not . isNaN) diffs
  assertApproxEqual (show thetas) gradientCheckingEps 0 diff


flattenTest = do
  theta <- initializeThetaIO nnt
  let theta' = flatten $ unflatten nnt theta
      norm = LA.norm_2 (theta - theta')
  assertApproxEqual "flatten" 1e-10 0 norm

nn_thetaSize = sum $ map (\(c, r) -> c*r) $ getThetaSizes nnt


xPredict = LA.matrix 2 [ -0.5, 0.5
                       , 0.2, -0.2
                       , 1, 1
                       , 1, 0
                       , 0, 0]
yExpected = LA.vector [1, 1, 0, 0, 1]

learnTest minMethod =
  let x1 = ML.addBiasDimension $ ML.mapFeatures 2 x
      nnt = makeTopology ((LA.cols x1) - 1) 2 [10]
      model = NeuralNetwork nnt
      xPredict1 = ML.addBiasDimension $ ML.mapFeatures 2 xPredict
      initTheta = initializeTheta 5191711 nnt
      (theta, _) = Opt.minimize minMethod model 1e-7 70 1 x1 y initTheta
      yPredicted = hypothesis model xPredict1 theta
  in do
    assertVector "" 0.01 yExpected yPredicted


tests = [ testGroup "thetaInitialization" [
            testCase "sizes" thetaSizeTest
            , testCase "theta total size" $ assertEqual "" nn_thetaSize (getThetaTotalSize nnt)
          ]
        , testGroup "gradient checking" [
            testCase "non-zero lambda" $ checkGradientTest 2
            , testCase "zero lambda" $ checkGradientTest 0
              ]
        , testGroup "flatten" [
            testCase "flatten" flattenTest
            ]
        , testGroup "learn" [
            testCase "BFGS" $ learnTest (Opt.BFGS2 0.03 0.7)
            ]
        ]
