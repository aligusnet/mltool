{-|
Module: MachineLearning.Random
Description: Random generation utility functions.
Copyright: (c) Alexander Ignatyev, 2017
License: BSD-3
Stability: experimental
Portability: POSIX

Randon generation uitility functions.
-}

module MachineLearning.Random
(
  sample
)

where

import Control.Monad (when, zipWithM_)
import Data.List (foldl')
import qualified System.Random as Rnd
import Data.Array (elems)
import qualified Data.Array.ST as A


-- | Samples `n` (given as a second parameter) values from `list` (given as a third parameter).
sample :: Rnd.RandomGen t => t -> Int -> [a] -> ([a], t)
sample gen n xs =
  let (rnds, gen') = genRandomListForSample gen n (length xs)
      (pre, post) = splitAt n xs
      ys = elems $ A.runSTArray $ do
        arr <- A.newListArray (0, n-1) pre
        zipWithM_ (\a r -> when (r < n) $ A.writeArray arr r a) post rnds
        return arr
  in (ys, gen')


-- | sample's helper function.
genRandomListForSample :: (Rnd.RandomGen t, Num a, Enum a, Rnd.Random a) => t -> a -> a -> ([a], t)
genRandomListForSample rndGen start finish = (reverse values, gen')
  where (firstValue, gen) = Rnd.randomR (0, start) rndGen
        generateNext (values, gen) f =
          let (value, gen') = Rnd.randomR (0, f) gen
          in (value:values, gen')
        (values, gen') = foldl' generateNext ([firstValue], gen) [start+1..finish-1]

