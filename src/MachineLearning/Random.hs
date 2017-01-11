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
  let rangeList = zip (repeat 0) [n..(length xs)-1]
      (rnds, gen') = randomsInRanges rangeList gen
      (pre, post) = splitAt n xs
      ys = elems $ A.runSTArray $ do
        arr <- A.newListArray (0, n-1) pre
        zipWithM_ (\a r -> when (r < n) $ A.writeArray arr r a) post rnds
        return arr
  in (ys, gen')


-- | Takes a list of ranges `(lo, hi)` and random generator `g`,
-- returns a list of random values uniformly distributed in the list of closed intervals [(lo, hi)].
randomsInRanges :: (Rnd.RandomGen t, Rnd.Random a) => [(a, a)] -> t -> ([a], t)
randomsInRanges rangeList gen = (reverse values, gen')
  where generateNext (values, g) interval =
          let (value, g') = Rnd.randomR interval g
          in (value:values, g')
        (values, gen') = foldl' generateNext ([], gen) rangeList

