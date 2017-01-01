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

import qualified Numeric.LinearAlgebra as LA
import qualified MachineLearning as ML
import MachineLearning.Regression (checkGradient)
import MachineLearning.Regression.Model
import MachineLearning.NeuralNetwork

(x, y) = ML.splitToXY dataset2

x1 = ML.addColumnOfOnes x
nnt = makeTopology (LA.cols x) 1 [10]
model = NeuralNetwork nnt


thetaSizeTest = do
  thetas <- initializeThetaList nnt
  let sizesActual = map LA.size thetas
      sizesExpected = getThetaSizes nnt
  assertEqual "theta sizes" sizesExpected sizesActual


checkGradientTest lambda = do
  thetas <- initializeTheta nnt
  let diff = checkGradient model lambda x1 y thetas 1e-2
  assertApproxEqual (show thetas) 2 0 diff


flattenTest = do
  thetas <- initializeThetaList nnt
  let thetas' = unflatten nnt $ flatten thetas
      norm = LA.norm_2 . LA.vector $ zipWith (\t1 t2 -> LA.norm_2 (t1-t2)) thetas thetas'
  assertApproxEqual "flatten" 1e-10 0 norm


nn_thetaSize = sum $ map (\(c, r) -> c*r) $ getThetaSizes nnt
onesTheta :: LA.Vector LA.R
onesTheta = LA.konst 0.1 (nn_thetaSize)

tests = [ testGroup "thetaInitialization" [
            testCase "sizes" thetaSizeTest
          ]
        , testGroup "gradient checking" [
            testCase "non-zero theta, non-zero lambda" $ assertApproxEqual "" 1 0 (checkGradient model 2 x1 y onesTheta 1e-3)
            , testCase "non-zero theta, zero lambda" $ assertApproxEqual "" 1 0 (checkGradient model 0 x1 y onesTheta 1e-3)
            , testCase "rand theta, non-zero lambda" $ checkGradientTest 2
            , testCase "rand theta, zero lambda" $ checkGradientTest 0
              ]
        , testGroup "flatten" [
            testCase "flatten" flattenTest
            ]
        ]
