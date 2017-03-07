module MachineLearning.Classification.BinaryTest
(
  tests
  , testOptPath
)

where

import Test.Framework (testGroup)
import Test.Framework.Providers.HUnit
import Test.HUnit
import Test.HUnit.Approx
import Test.HUnit.Plus

import MachineLearning.Types
import qualified Data.Vector as V
import qualified Numeric.LinearAlgebra as LA

import MachineLearning.DataSets (dataset2)

import qualified MachineLearning as ML
import MachineLearning.Classification.Binary

(x, y) = ML.splitToXY dataset2


processX muSigma x = ML.addBiasDimension $ ML.featureNormalization muSigma $ ML.mapFeatures 6 x

muSigma = ML.meanStddev (ML.mapFeatures 6 x)
x1 = processX muSigma x
zeroTheta = LA.konst 0 (LA.cols x1)

xPredict = LA.matrix 2 [ -0.5, 0.5
                       , 0.2, -0.2
                       , 1, 1
                       , 1, 0
                       , 0, 0
                       , 0, 1]
xPredict1 = processX muSigma xPredict
yExpected = LA.vector [1, 1, 0, 0, 1, 0]

eps = 0.0001


-- Binary

(thetaCGFR, optPathCGFR) = learn (ConjugateGradientFR 0.1 0.1) eps 50 (L2 0.5) x1 y zeroTheta
(thetaCGPR, optPathCGPR) = learn (ConjugateGradientPR 0.1 0.1) eps 50 (L2 0.5) x1 y zeroTheta
(thetaBFGS, optPathBFGS) = learn (BFGS2 0.1 0.1) eps 50 (L2 0.5) x1 y zeroTheta


isInDescendingOrder :: V.Vector Double -> Bool
isInDescendingOrder lst = V.and . snd . V.unzip $ V.scanl (\(prev, _) current -> (current, prev-current > (-0.001))) (1/0, True) lst

testOptPath optPath = do
  let js = V.convert $ (LA.toColumns optPath) !! 1
  assertBool ("non-increasing errors: " ++ show js) $ isInDescendingOrder js

testAccuracyBinary theta eps = do
  let yPredicted = predict x1 theta
      accuracy = calcAccuracy y yPredicted
  assertApproxEqual "" eps 1 accuracy

tests = [
  testGroup "learn" [
      testCase "Conjugate Gradient FR" $ assertVector "" 0.01 yExpected (predict xPredict1 thetaCGFR)
      , testCase "Conjugate Gradient PR" $ assertVector "" 0.01 yExpected (predict xPredict1 thetaCGPR)
      , testCase "BFGS" $ assertVector "" 0.01 yExpected (predict xPredict1 thetaBFGS)
      ]
  , testGroup "optPath" [
      testCase "Conjugate Gradient FR" $ testOptPath optPathCGFR
      , testCase "Conjugate Gradient PR" $ testOptPath optPathCGPR
      , testCase "BFGS" $ testOptPath optPathBFGS
      ]
    , testGroup "accuracy" [
        testCase "Conjugate Gradient FR" $ testAccuracyBinary thetaCGFR 0.2
        , testCase "Conjugate Gradient PR" $ testAccuracyBinary thetaCGPR 0.2
        , testCase "BFGS" $ testAccuracyBinary thetaBFGS 0.2
        ]
  ]
