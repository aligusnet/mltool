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
import qualified System.Random as Rnd

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

tests = [ testCase "sample" $ sampleTest
        ]
