module MachineLearning.Regression.LeastSquaresTest
(
  tests
)

where

import Test.Framework (testGroup)
import Test.Framework.Providers.HUnit
import Test.HUnit
import Test.HUnit.Approx
import Test.HUnit.Plus

import MachineLearning.Regression.DataSets (dataset1)

import qualified Numeric.LinearAlgebra as LA
import qualified MachineLearning as ML
import MachineLearning.Regression (checkGradient)
import MachineLearning.Regression.Model
import MachineLearning.Regression.LeastSquares

(x, y) = ML.splitToXY dataset1

x1 = ML.addColumnOfOnes x
initialTheta :: LA.Vector LA.R
initialTheta = LA.konst 1000 (LA.cols x1)
zeroTheta :: LA.Vector LA.R
zeroTheta = LA.konst 0 (LA.cols x1)

tests = [ testGroup "model" [
            testCase "cost, lambda = 0"      $ assertApproxEqual "" 1e-5 1.6190245331702874e12 (cost LeastSquares 0 x1 y initialTheta)
            , testCase "cost, lambda = 1"    $ assertApproxEqual "" 1e-5 1.619024554446883e12 (cost LeastSquares 1 x1 y initialTheta)
            , testCase "cost, lambda = 1000" $ assertApproxEqual "" 1e-5 1.619045809766032e12 (cost LeastSquares 1000 x1 y initialTheta)
            , testCase "gradient, lambda = 0" $ assertVector 1e-5 gradient_l0 (gradient LeastSquares 0 x1 y initialTheta)
            , testCase "gradient, lambda = 1" $ assertVector 1e-5 gradient_l1 (gradient LeastSquares 1 x1 y initialTheta)
            , testCase "gradient, lambda = 1000" $ assertVector 1e-5 gradient_l1000 (gradient LeastSquares 1000 x1 y initialTheta)
            ]
          , testGroup "gradient checking" [
              testCase "non-zero theta, non-zero lambda" $ assertBool "" $ (checkGradient LeastSquares 2 x1 y initialTheta 1e-4) < 10
              , testCase "zero theta, non-zero lambda" $ assertBool "" $ (checkGradient LeastSquares 2 x1 y zeroTheta 1e-4) < 1
              , testCase "non-zero theta, zero lambda" $ assertBool "" $ (checkGradient LeastSquares 0 x1 y initialTheta 1e-4) < 5
              , testCase "zero theta, zero lambda" $ assertBool "" $ (checkGradient LeastSquares 0 x1 y zeroTheta 1e-4) < 1
              ]
        ]

gradient_l0    = LA.vector [1664438.4042553192,3.865303999468085e9,5567440.808510638]
gradient_l1    = LA.vector [1664438.4042553192,3.865304020744681e9,5567462.085106383]
gradient_l1000 = LA.vector [1664438.4042553192,3.86532527606383e9, 5588717.404255319]
