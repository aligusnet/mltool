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
import qualified System.Random as Rnd
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
  let gen = Rnd.mkStdGen 10171
      (clusters, !gen') = kmeans 10 x k gen
  print clusters
  assertEqual "number of clusters" expectedK (V.length clusters)


tests = [testGroup "kmeans" [
            testCase "perfect clusters, k = 2" $ testKmeans x1 2 2
            , testCase "perfect clusters, k = 3" $ testKmeans x1 3 2
            , testCase "good clusters, k = 2" $ testKmeans x2 2 2
            , testCase "good clusters, k = 3" $ testKmeans x2 3 3
            , testCase "good clusters, k = 4" $ testKmeans x2 4 4
            ]
        ]
