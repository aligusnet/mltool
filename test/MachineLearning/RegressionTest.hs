module MachineLearning.RegressionTest
(
  tests
)

where

import Test.Framework (testGroup)
import Test.Framework.Providers.HUnit
import Test.HUnit
import Test.HUnit.Approx
import Test.HUnit.Plus

import qualified Numeric.LinearAlgebra as LA

import MachineLearning.Regression.DataSets (dataset1)

import qualified MachineLearning as ML
import MachineLearning.Regression

(x, y) = ML.splitToXY dataset1

muSigma = ML.meanStddev x
xNorm = ML.featureNormalization muSigma x
x1 = ML.addColumnOfOnes xNorm
zeroTheta = LA.konst 0 (LA.cols x1)

xPredict = LA.matrix 2 [1650, 3]
xPredict1 = ML.addColumnOfOnes $ ML.featureNormalization muSigma xPredict

thetaNE = ML.normalEquation (ML.addColumnOfOnes x) y
yExpected = hypothesis LeastSquares (ML.addColumnOfOnes xPredict) thetaNE

eps = 0.0001
(thetaGD, _) = minimize (GradientDescent 0.01) LeastSquares eps 5000 0 x1 y zeroTheta
(thetaCGFR, _) = minimize (ConjugateGradientFR 0.1 0.1) LeastSquares eps 1500 0 x1 y zeroTheta
(thetaCGPR, _) = minimize (ConjugateGradientPR 0.1 0.1) LeastSquares eps 1500 0 x1 y zeroTheta
(thetaBFGS, _) = minimize (BFGS2 0.1 0.1) LeastSquares eps 1500 0 x1 y zeroTheta


tests = [ testGroup "minimize" [
            testCase "Gradient Descent" $ assertVector 0.01 yExpected (hypothesis LeastSquares xPredict1 thetaGD)
            , testCase "BFGS" $ assertVector 0.01 yExpected (hypothesis LeastSquares xPredict1 thetaBFGS)
            , testCase "Conjugate Gradient FR" $ assertVector 0.01 yExpected (hypothesis LeastSquares xPredict1 thetaCGFR)
            , testCase "Conjugate Gradient PR" $ assertVector 0.01 yExpected (hypothesis LeastSquares xPredict1 thetaCGPR)
            ]
        ]
