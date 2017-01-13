module MachineLearning.ClassificationTest
(
  tests
)

where

import Test.Framework (testGroup)
import Test.Framework.Providers.HUnit
import Test.HUnit
import Test.HUnit.Approx
import Test.HUnit.Plus

import Types
import qualified Data.Vector as V
import qualified Numeric.LinearAlgebra as LA

import MachineLearning.Regression.DataSets (dataset2)

import qualified MachineLearning as ML
import MachineLearning.Classification

(x, y) = ML.splitToXY dataset2


processX muSigma x = ML.addColumnOfOnes $ ML.featureNormalization muSigma $ ML.mapFeatures 6 x

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

(thetaCGFR, optPathCGFR) = learnBinary (ConjugateGradientFR 0.1 0.1) eps 50 0.5 x1 y zeroTheta
(thetaCGPR, optPathCGPR) = learnBinary (ConjugateGradientPR 0.1 0.1) eps 50 0.5 x1 y zeroTheta
(thetaBFGS, optPathBFGS) = learnBinary (BFGS2 0.1 0.1) eps 50 0.5 x1 y zeroTheta

-- Multi
ym = processOutputMulti 2 y
zeroThetam = replicate (length ym) zeroTheta
(thetaGDm, optPathGDm) = learnMulti (GradientDescent 0.01) eps 250 0.5 x1 ym zeroThetam
(thetaCGFRm, optPathCGFRm) = learnMulti (ConjugateGradientFR 0.1 0.1) eps 50 0.5 x1 ym zeroThetam
(thetaCGPRm, optPathCGPRm) = learnMulti (ConjugateGradientPR 0.1 0.1) eps 50 0.5 x1 ym zeroThetam
(thetaBFGSm, optPathBFGSm) = learnMulti (BFGS2 0.1 0.1) eps 50 0.5 x1 ym zeroThetam


isInDescendingOrder :: V.Vector Double -> Bool
isInDescendingOrder lst = V.and . snd . V.unzip $ V.scanl (\(prev, _) current -> (current, prev-current > (-0.0001))) (1/0, True) lst

testOptPath optPath = do
  let js = V.convert $ (LA.toColumns optPath) !! 1
  assertBool "non-increasing errors" $ isInDescendingOrder js

testAccuracyBinary theta eps = do
  let yPredicted = predictBinary x1 theta
      accuracy = calcAccuracy y yPredicted
  assertApproxEqual "" eps 1 accuracy

tests = [ testGroup "binary" testsBinary
          , testGroup "multi" testsMulti
        ]

testsBinary = [
  testGroup "learn" [
      testCase "Conjugate Gradient FR" $ assertVector "" 0.01 yExpected (predictBinary xPredict1 thetaCGFR)
      , testCase "Conjugate Gradient PR" $ assertVector "" 0.01 yExpected (predictBinary xPredict1 thetaCGPR)
      , testCase "BFGS" $ assertVector "" 0.01 yExpected (predictBinary xPredict1 thetaBFGS)
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

testsMulti = [
  testGroup "learn" [
      testCase "Gradient Descent" $ assertVector "" 0.01 yExpected (predictMulti xPredict1 thetaGDm)
      , testCase "Conjugate Gradient FR" $ assertVector "" 0.01 yExpected (predictMulti xPredict1 thetaCGFRm)
      , testCase "Conjugate Gradient PR" $ assertVector "" 0.01 yExpected (predictMulti xPredict1 thetaCGPRm)
      , testCase "BFGS" $ assertVector "" 0.01 yExpected (predictMulti xPredict1 thetaBFGSm)
      ]
  , testGroup "optPath" [
      testCase "Gradient Descent" $ mapM_ testOptPath optPathGDm
      , testCase "Conjugate Gradient FR" $ mapM_ testOptPath optPathCGFRm
      , testCase "Conjugate Gradient PR" $ mapM_ testOptPath optPathCGPRm
      , testCase "BFGS" $ mapM_ testOptPath optPathBFGSm
      ]
  ]
