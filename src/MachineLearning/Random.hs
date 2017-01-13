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
  , sampleM
)

where

import Control.Monad (when)
import System.Random (RandomGen)
import qualified Control.Monad.ST as ST
import qualified Data.Vector as V
import qualified Data.Vector.Mutable as MV
import qualified Control.Monad.Random as RndM


-- | Samples `n` (given as a second parameter) values from `list` (given as a third parameter).
sample :: (RandomGen g, Show a) => g -> Int -> V.Vector a -> (V.Vector a, g)
sample gen n xs = RndM.runRand (sampleM n xs) gen


-- | Samples `n` (given as a second parameter) values from `list` (given as a third parameter) inside RandomMonad.
sampleM :: (RandomGen g, Show a) => Int -> V.Vector a -> RndM.Rand g (V.Vector a)
sampleM n xs = do  -- Random Monad starts
  let rangeList = V.fromList $ zip (repeat 0) [n..(length xs)-1]
  rnds <- randomsInRangesM rangeList
  let (pre, post) = V.splitAt n xs
  let ys = ST.runST $ do  -- ST Monad starts
        mv <- V.thaw pre
        V.zipWithM_ (\val r -> when (r < n) $ MV.write mv (mod r n) val) post rnds
        V.unsafeFreeze mv
  return ys
  

-- | Takes a list of ranges `(lo, hi)`,
-- returns a list of random values uniformly distributed in the list of closed intervals [(lo, hi)].
randomsInRangesM :: (RndM.RandomGen g, RndM.Random a) => V.Vector (a, a) -> RndM.Rand g (V.Vector a)
randomsInRangesM rangeList = mapM RndM.getRandomR rangeList
