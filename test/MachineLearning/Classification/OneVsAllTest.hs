module MachineLearning.Classification.OneVsAllTest
(
  tests
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
import MachineLearning.Classification.OneVsAll
import MachineLearning.Classification.BinaryTest (testOptPath)

(x, y) = ML.splitToXY dataset2


processX muSigma x = ML.addBiasDimension $ ML.featureNormalization muSigma $ ML.mapFeatures 6 x

muSigma = ML.meanStddev (ML.mapFeatures 6 x)
x1 = processX muSigma x
zeroTheta :: Vector
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


zeroThetam = replicate 2 zeroTheta
(thetaGDm, optPathGDm) = learn (GradientDescent 0.001) eps 200 0.5 2 x1 y zeroThetam
(thetaCGFRm, optPathCGFRm) = learn (ConjugateGradientFR 0.1 0.1) eps 50 0.5 2 x1 y zeroThetam
(thetaCGPRm, optPathCGPRm) = learn (ConjugateGradientPR 0.1 0.1) eps 50 0.5 2 x1 y zeroThetam
(thetaBFGSm, optPathBFGSm) = learn (BFGS2 0.1 0.1) eps 50 0.5 2 x1 y zeroThetam


tests = [
  testGroup "learn" [
      testCase "Gradient Descent" $ assertVector "" 0.01 yExpected (predict xPredict1 thetaGDm)
      , testCase "Conjugate Gradient FR" $ assertVector "" 0.01 yExpected (predict xPredict1 thetaCGFRm)
      , testCase "Conjugate Gradient PR" $ assertVector "" 0.01 yExpected (predict xPredict1 thetaCGPRm)
      , testCase "BFGS" $ assertVector "" 0.01 yExpected (predict xPredict1 thetaBFGSm)
      ]
  , testGroup "optPath" [
      testCase "Gradient Descent" $ mapM_ testOptPath optPathGDm
      , testCase "Conjugate Gradient FR" $ mapM_ testOptPath optPathCGFRm
      , testCase "Conjugate Gradient PR" $ mapM_ testOptPath optPathCGPRm
      , testCase "BFGS" $ mapM_ testOptPath optPathBFGSm
      ]
  ]
