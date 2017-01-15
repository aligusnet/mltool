module MachineLearning.PCATest
(
  tests
)

where

import Test.Framework (testGroup)
import Test.Framework.Providers.HUnit
import Test.HUnit
import Test.HUnit.Approx

import qualified Numeric.LinearAlgebra as LA

import MachineLearning.PCA

getDimReducerSmokeTest =
  let nFeatures = 4
      nExamples = 7
      m = LA.matrix nFeatures [1 .. fromIntegral $ nFeatures*nExamples]
      m10 = m * 10
      (reduceDims, retainedVariance, mReduced) = getDimReducer m 2
      m10Reduced = reduceDims m10
  in do
    assertEqual "dimension equality (getDimReducer)" (LA.cols mReduced) 2
    assertEqual "dimension equality (reduceDims)" (LA.cols m10Reduced) 2
    assertApproxEqual "retained variance" 1e-10 1 retainedVariance

getDimReducer_rvSmokeTest rv=
  let nFeatures = 4
      nExamples = 7
      m = LA.matrix nFeatures [1 .. fromIntegral $ nFeatures*nExamples]
      m10 = m * 10
      (reduceDims, k, mReduced) = getDimReducer_rv m rv
      m10Reduced = reduceDims m10
  in do
    assertEqual "dimension equality (getDimReducer_rv)" (LA.cols mReduced) k
    assertEqual "dimension equality (reduceDims_rv)" (LA.cols m10Reduced) k
    assertBool "reduced number of dimensions" $ k <= nFeatures

tests = [ testGroup "smoke test" [
            testCase "getDimReducer" getDimReducerSmokeTest
            , testCase "getDimReducer_rv, rv = 0" $ getDimReducer_rvSmokeTest 0
            , testCase "getDimReducer_rv, rv = 0.5" $ getDimReducer_rvSmokeTest 0.5
            , testCase "getDimReducer_rv, rv = 1" $ getDimReducer_rvSmokeTest 1
            , testCase "getDimReducer_rv, rv = 2" $ getDimReducer_rvSmokeTest 2
            ]
        ]
