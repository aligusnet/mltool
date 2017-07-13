module MachineLearning.SoftmaxClassifierTest
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
import MachineLearning.SoftmaxClassifier

(x, y) = ML.splitToXY dataset2

model = MultiClass (Softmax 2)

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


checkSoftmaxGradient theta eps lambda = minimum . take 5 $ map check [eps, eps+0.001 ..]
  where check e = checkGradient model lambda x1 y theta e
  

gradientCheckingEps :: Double
gradientCheckingEps = 3e-2

eps = 0.000001

initialTheta = LA.konst 0.001 (2 * LA.cols x2)
(thetaGD, optPathGD) = minimize (GradientDescent 0.0005) model eps 150 (L2 1) x2 y initialTheta
(thetaCGFR, optPathCGFR) = minimize (ConjugateGradientFR 0.05 0.2) model eps 30 (L2 1) x2 y initialTheta
(thetaCGPR, optPathCGPR) = minimize (ConjugateGradientPR 0.05 0.3) model eps 30 (L2 1) x2 y initialTheta

showOptPath optPath = show $  (LA.toColumns optPath) !! 1


tests = [  testGroup "gradient checking" [
              testCase "non-zero theta, non-zero lambda" $ assertApproxEqual "" gradientCheckingEps 0 (checkSoftmaxGradient onesTheta 1e-3 (L2 2))
              , testCase "zero theta, non-zero lambda" $ assertApproxEqual "" gradientCheckingEps 0 (checkSoftmaxGradient zeroTheta 1e-3 (L2 2))
              , testCase "non-zero theta, zero lambda" $ assertApproxEqual "" gradientCheckingEps 0 (checkSoftmaxGradient onesTheta 1e-3 (L2 0))
              , testCase "zero theta, zero lambda" $ assertApproxEqual "" gradientCheckingEps 0 (checkSoftmaxGradient zeroTheta 1e-3 (L2 0))
              , testCase "non-zero theta, no reg" $ assertApproxEqual "" gradientCheckingEps 0 (checkSoftmaxGradient onesTheta 1e-3 RegNone)
              , testCase "zero theta, no reg" $ assertApproxEqual "" gradientCheckingEps 0 (checkSoftmaxGradient zeroTheta 1e-3 RegNone)
              ]
           
        , testGroup "learn" [
            testCase "Gradient Descent" $ assertVector (showOptPath optPathGD) 0.01 yExpected (hypothesis model xPredict2 thetaGD)
            , testCase "Conjugate Gradient FR" $ assertVector (showOptPath optPathCGFR) 0.01 yExpected (hypothesis model xPredict2 thetaCGFR)
            , testCase "Conjugate Gradient PR" $ assertVector (showOptPath optPathCGPR) 0.01 yExpected (hypothesis model xPredict2 thetaCGPR)
            ]
        ]

