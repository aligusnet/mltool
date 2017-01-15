module MachineLearning.RandomTest
(
  tests
)

where

import Test.Framework (testGroup)
import Test.Framework.Providers.HUnit
import Test.HUnit
import Test.HUnit.Approx
import Test.HUnit.Plus

import MachineLearning.Random

import Data.List (nub)
import Control.Monad (foldM_)
import qualified Data.Vector as V
import qualified Data.Vector.Storable as SV
import qualified Numeric.LinearAlgebra as LA
import qualified System.Random as Rnd
import qualified Control.Monad.Random as RndM

sampleTest = do
  gen <- Rnd.newStdGen
  foldM_ sampleTestIter gen [1..25]

sampleTestIter gen i =
  let xs = V.fromList [1..100]
      n = 10 + i
      (ys, gen') = sample gen n xs
  in do
    assertEqual "uniqness" (V.length xs) (length . nub $ V.toList xs) 
    assertEqual "length" n (V.length ys)
    assertBool "maximum" $ (V.maximum ys) <= (V.maximum xs)
    assertBool "minimum" $ (V.minimum ys) >= (V.minimum xs)
    assertEqual "uniqness of elements" n (length . nub $ V.toList ys) 
    return gen'


randomRListTest = mapM_ (randomRListTestIter ((-1000, 1000)::(Int, Int))) [10..30]

randomRListTestIter range@(lo, hi) len = do
  rndList <- RndM.evalRandIO (getRandomRListM len range)
  assertEqual "length" len (length rndList)
  assertBool "minimum" $ lo <= (minimum rndList)
  assertBool "maximum" $ hi >= (maximum rndList)


randomRVectorTest = mapM_ (randomRVectorTestIter ((-2, 2))) [10..30]

randomRVectorTestIter range@(lo, hi) len = do
  rndVector <- RndM.evalRandIO (getRandomRVectorM len range)
  assertEqual "length" len (SV.length rndVector)
  assertBool "minimum" $ lo <= (SV.minimum rndVector)
  assertBool "maximum" $ hi >= (SV.maximum rndVector)


randomRMatrixTest = mapM_ (randomRMatrixTestIter ((-2, 2))) $ zip [10..15] [12..17]

randomRMatrixTestIter range@(lo, hi) (rows, cols) = do
  rndMatrix <- RndM.evalRandIO (getRandomRMatrixM rows cols range)
  assertEqual "rows" rows (LA.rows rndMatrix)
  assertEqual "columns" cols (LA.cols rndMatrix)
  assertBool "minimum" $ lo <= (LA.minElement rndMatrix)
  assertBool "maximum" $ hi >= (LA.maxElement rndMatrix)


tests = [ testCase "sample" sampleTest
        , testCase "getRandomRList" randomRListTest
        , testCase "getRandomRVector" randomRVectorTest
        , testCase "getRandomRMatrix" randomRMatrixTest
        ]
