{-# LANGUAGE BangPatterns #-}
module MachineLearning.ClusteringTest
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
import Numeric.LinearAlgebra ((><))
import qualified Control.Monad.Random as RndM
import MachineLearning.Clustering

x1 :: Matrix
x1 = (7><5) [ 1, 2, 3, 4, 5
            , 1, 2, 3, 4, 5
            , 1, 2, 3, 4, 5
            , 7, 4, 3, 2, 1
            , 7, 4, 3, 2, 1
            , 1, 2, 3, 4, 5
            , 1, 2, 3, 4, 5]

x2 :: Matrix
x2 = (7><5) [ 1.1, 2, 3, 4, 5
            , 1, 2, 4, 4, 5
            , 1, 2, 3, 4, 5
            , 5, 4, 3, 2, 1
            , 7, 4, 3, 2, 1
            , 1, 2, 3, 4, 5
            , 0.5, 2, 3, 4, 5]


testKmeans x k expectedK = do
  let gen = RndM.mkStdGen 10171
      clusters = RndM.evalRand (kmeans 10 x k) gen
  assertEqual "number of clusters" expectedK (V.length clusters)

isInDescendingOrder :: [Double] -> Bool
isInDescendingOrder lst = and . snd . unzip $ scanl (\(prev, _) current -> (current, prev-current > (-0.001))) (1/0, True) lst

testDescOrderOfCostValues = do
  let gen = RndM.mkStdGen 10171
      samples = V.fromList $ LA.toRows x1
      (clusters, js) = RndM.evalRand (kmeansIterM samples 3 1) gen
  assertBool "" (isInDescendingOrder js)

tests = [testGroup "kmeans" [
            testCase "perfect clusters, k = 2" $ testKmeans x1 2 2
            , testCase "perfect clusters, k = 3" $ testKmeans x1 3 2
            , testCase "good clusters, k = 2" $ testKmeans x2 2 2
            , testCase "good clusters, k = 3" $ testKmeans x2 3 3
            , testCase "good clusters, k = 4" $ testKmeans x2 4 4
            , testCase "descending order" testDescOrderOfCostValues
            ]
        ]
