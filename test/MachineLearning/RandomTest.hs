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
import qualified System.Random as Rnd

sampleTest = do
  gen <- Rnd.newStdGen
  foldM_ sampleTestIter gen [1..25]

sampleTestIter gen i =
  let xs = [1..100]
      n = 10 + i
      (ys, gen') = sample gen n xs
  in do
    assertEqual "length" n (length ys)
    assertBool "maximum" $ (maximum ys) <= (maximum xs)
    assertBool "minimum" $ (minimum ys) >= (minimum xs)
    assertEqual "uniqness of elements" n (length $ nub ys) 
    return gen'

tests = [ testCase "sample" $ sampleTest
        ]
