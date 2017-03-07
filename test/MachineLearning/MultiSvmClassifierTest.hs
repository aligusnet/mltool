module MachineLearning.MultiSvmClassifierTest
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
import MachineLearning.Model
import MachineLearning.MultiSvmClassifier

(x, y) = ML.splitToXY dataset2

model = MultiClass (MultiSvm 1 2)

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


gradientCheckingEps :: Double
gradientCheckingEps = 1e-3

eps = 0.0001

zeroTheta2 = LA.konst 0 (2 * LA.cols x2)
(thetaGD, _) = minimize (GradientDescent 0.001) model eps 150 (L2 1) x2 y zeroTheta2
(thetaCGFR, _) = minimize (ConjugateGradientFR 0.1 0.1) model eps 30 (L2 0.5) x2 y zeroTheta2
(thetaCGPR, _) = minimize (ConjugateGradientPR 0.1 0.1) model eps 30 (L2 0.5) x2 y zeroTheta2
(thetaBFGS, _) = minimize (BFGS2 0.1 0.1) model eps 30 (L2 0.5) x2 y zeroTheta2


checkGradientTest lambda theta eps = do
  let diffs = take 5 $ map (\e -> checkGradient model lambda x1 y theta e) [1e-3, 1.1e-3 ..]
      diff = minimum $ filter (not . isNaN) diffs
  assertApproxEqual "" eps 0 diff


tests = [  testGroup "gradient checking" [
            testCase "non-zero theta, non-zero lambda" $ checkGradientTest (L2 2) onesTheta 3e-2
              , testCase "zero theta, non-zero lambda" $ checkGradientTest (L2 2) zeroTheta gradientCheckingEps
              , testCase "non-zero theta, zero lambda" $ checkGradientTest (L2 0) onesTheta gradientCheckingEps
              , testCase "zero theta, zero lambda" $ checkGradientTest (L2 0) zeroTheta gradientCheckingEps
              , testCase "non-zero theta, no reg" $ checkGradientTest RegNone onesTheta gradientCheckingEps
              , testCase "zero theta, no reg" $ checkGradientTest RegNone zeroTheta gradientCheckingEps
              ]
        , testGroup "learn" [
            testCase "Gradient Descent" $ assertVector "" 0.01 yExpected (hypothesis model xPredict2 thetaGD)
            , testCase "Conjugate Gradient FR" $ assertVector "" 0.01 yExpected (hypothesis model xPredict2 thetaCGFR)
            , testCase "Conjugate Gradient PR" $ assertVector "" 0.01 yExpected (hypothesis model xPredict2 thetaCGPR)
            , testCase "BFGS" $ assertVector "" 0.01 yExpected (hypothesis model xPredict2 thetaBFGS)
            ]
        ]

