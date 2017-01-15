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

import MachineLearning.Regression.DataSets (dataset2)

import qualified Control.Monad.Random as RndM
import qualified Numeric.LinearAlgebra as LA
import qualified MachineLearning as ML
import qualified MachineLearning.Regression as MLR
import MachineLearning.NeuralNetwork

(x, y) = ML.splitToXY dataset2

x1 = ML.addColumnOfOnes x
nnt = makeTopology (LA.cols x) 2 [10]
model = NeuralNetwork nnt

gradientCheckingEps = 10


thetaSizeTest = do
  thetas <- RndM.evalRandIO $ initializeThetaListM nnt
  let sizesActual = map LA.size thetas
      sizesExpected = getThetaSizes nnt
  assertEqual "theta sizes" sizesExpected sizesActual


checkGradientTest lambda = do
  let thetas = initializeTheta 1511197 nnt
      diff = MLR.checkGradient model lambda x1 y thetas 1e-2
  assertApproxEqual (show thetas) gradientCheckingEps 0 diff


flattenTest = do
  theta <- initializeThetaIO nnt
  let theta' = flatten $ unflatten nnt theta
      norm = LA.norm_2 (theta - theta')
  assertApproxEqual "flatten" 1e-10 0 norm

nn_thetaSize = sum $ map (\(c, r) -> c*r) $ getThetaSizes nnt
onesTheta :: LA.Vector LA.R
onesTheta = LA.konst 0.01 (nn_thetaSize)


xPredict = LA.matrix 2 [ -0.5, 0.5
                       , 0.2, -0.2
                       , 1, 1
                       , 1, 0
                       , 0, 0
                       , 0, 1]
yExpected = LA.vector [1, 1, 0, 0, 1, 0]

learnTest minMethod =
  let x1 = ML.addColumnOfOnes $ ML.mapFeatures 2 x
      nnt = makeTopology ((LA.cols x1) - 1) 2 [10]
      model = NeuralNetwork nnt
      xPredict1 = ML.addColumnOfOnes $ ML.mapFeatures 2 xPredict
      initTheta = initializeTheta 5191711 nnt
      (theta, _) = MLR.minimize minMethod model 1e-7 50 1 x1 y initTheta
      yPredicted = MLR.hypothesis model xPredict1 theta
  in do
    assertVector "" 0.01 yExpected yPredicted
      

tests = [ testGroup "thetaInitialization" [
            testCase "sizes" thetaSizeTest
            , testCase "theta total size" $ assertEqual "" nn_thetaSize (getThetaTotalSize nnt)
          ]
        , testGroup "gradient checking" [
            testCase "non-zero theta, non-zero lambda" $ assertApproxEqual "" gradientCheckingEps 0 (MLR.checkGradient model 2 x1 y onesTheta 1e-3)
            , testCase "non-zero theta, zero lambda" $ assertApproxEqual "" gradientCheckingEps 0 (MLR.checkGradient model 0 x1 y onesTheta 1e-3)
            , testCase "rand theta, non-zero lambda" $ checkGradientTest 2
            , testCase "rand theta, zero lambda" $ checkGradientTest 0
              ]
        , testGroup "flatten" [
            testCase "flatten" flattenTest
            ]
        , testGroup "learn" [
            testCase "BFGS" $ learnTest (MLR.BFGS2 0.03 0.7)
            ]
        ]
