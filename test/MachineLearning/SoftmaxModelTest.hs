module MachineLearning.SoftmaxModelTest
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

import qualified Numeric.LinearAlgebra as LA
import qualified MachineLearning as ML
import MachineLearning.Optimization
import MachineLearning.SoftmaxModel

(x, y) = ML.splitToXY dataset2

model = Softmax 2

x1 = ML.addBiasDimension x
onesTheta :: LA.Vector LA.R
onesTheta = LA.konst 1 (2 * LA.cols x1)
zeroTheta :: LA.Vector LA.R
zeroTheta = LA.konst 0 (2 * LA.cols x1)

processX muSigma x = ML.addBiasDimension $ ML.featureNormalization muSigma $ ML.mapFeatures 6 x

muSigma = ML.meanStddev (ML.mapFeatures 6 x)
x2 = processX muSigma x


xPredict = LA.matrix 2 [ -0.5, 0.5
                       , 0.2, -0.2
                       , 1, 1
                       , 1, 0
                       , 0, 0
                       , 0, 1]
xPredict2 = processX muSigma xPredict
yExpected = LA.vector [1, 1, 0, 0, 1, 0]


checkSoftmaxGradient theta eps = minimum . take 5 . filter (not . isNaN) $ map check [eps, eps+0.001 ..]
  where check e = checkGradient model 2 x1 y theta e
  

gradientCheckingEps :: Double
gradientCheckingEps = 3e-2

eps = 0.0001

zeroTheta2 = LA.konst 0 (2 * LA.cols x2)
(thetaGD, _) = minimize (GradientDescent 0.001) model eps 200 0.5 x2 y zeroTheta2
(thetaCGFR, _) = minimize (ConjugateGradientFR 0.1 0.1) model eps 50 0.5 x2 y zeroTheta2
(thetaCGPR, _) = minimize (ConjugateGradientPR 0.1 0.1) model eps 50 0.5 x2 y zeroTheta2


tests = [  testGroup "gradient checking" [
              testCase "non-zero theta, non-zero lambda" $ assertApproxEqual "" gradientCheckingEps 0 (checkSoftmaxGradient onesTheta 1e-3)
              , testCase "zero theta, non-zero lambda" $ assertApproxEqual "" gradientCheckingEps 0 (checkSoftmaxGradient zeroTheta 1e-3)
              , testCase "non-zero theta, zero lambda" $ assertApproxEqual "" gradientCheckingEps 0 (checkSoftmaxGradient onesTheta 1e-3)
              , testCase "zero theta, zero lambda" $ assertApproxEqual "" gradientCheckingEps 0 (checkSoftmaxGradient zeroTheta 1e-3)
              ]
        , testGroup "learn" [
            testCase "Gradient Descent" $ assertVector "" 0.01 yExpected (hypothesis model xPredict2 thetaGD)
            , testCase "Conjugate Gradient FR" $ assertVector "" 0.01 yExpected (hypothesis model xPredict2 thetaCGFR)
            , testCase "Conjugate Gradient PR" $ assertVector "" 0.01 yExpected (hypothesis model xPredict2 thetaCGPR)
            ]
        ]

