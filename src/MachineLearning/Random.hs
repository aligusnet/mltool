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
  , getRandomRListM
  , getRandomRVectorM
  , getRandomRMatrixM
)

where

import Control.Monad (when)
import System.Random (RandomGen, Random)
import qualified Control.Monad.ST as ST
import qualified Data.Vector as V
import qualified Data.Vector.Storable as SV
import Numeric.LinearAlgebra ((><))
import qualified Data.Vector.Mutable as MV
import qualified Control.Monad.Random as RndM
import MachineLearning.Types (R, Vector, Matrix)


-- | Samples `n` (given as a second parameter) values from `list` (given as a third parameter).
sample :: RandomGen g => g -> Int -> V.Vector a -> (V.Vector a, g)
sample gen n xs = RndM.runRand (sampleM n xs) gen


-- | Samples `n` (given as a second parameter) values from `list` (given as a third parameter) inside RandomMonad.
sampleM :: RandomGen g => Int -> V.Vector a -> RndM.Rand g (V.Vector a)
sampleM n xs = do  -- Random Monad starts
  let rangeList = V.fromList $ zip (repeat 0) [n..(length xs)-1]
  rnds <- randomsInRangesM rangeList
  let (pre, post) = V.splitAt n xs
  let ys = ST.runST $ do  -- ST Monad starts
        mv <- V.thaw pre
        V.zipWithM_ (\val r -> when (r < n) $ MV.write mv (mod r n) val) post rnds
        V.unsafeFreeze mv
  return ys


-- | Returns a list of random values distributed in a closed interval `range`
getRandomRListM :: (RandomGen g, Random a) =>
                   Int             -- ^ list's lengths
                   -> (a, a)       -- ^ range
                   -> RndM.Rand g [a]          -- ^ list of random values inside RandomMonad
getRandomRListM size range = mapM (\_ -> RndM.getRandomR range) [1..size]


-- | Returns a vector of random values distributed in a closed interval `range`
getRandomRVectorM :: RandomGen g =>
                    Int                     -- ^ vector's length
                    -> (R, R)               -- ^ range
                    -> RndM.Rand g Vector   -- vector of randon values inside RandomMonad
getRandomRVectorM size range = SV.fromList <$> getRandomRListM size range


-- | Returns a matrix of random values distributed in a closed interval `range`
getRandomRMatrixM :: RandomGen g =>
                    Int                     -- ^ number of rows
                    -> Int                  -- ^ number of columns
                    -> (R, R)               -- ^ range
                    -> RndM.Rand g Matrix   -- vector of randon values inside RandomMonad
getRandomRMatrixM r c range = (r><c) <$> getRandomRListM (r*c) range


-- | Takes a list of ranges `(lo, hi)`,
-- returns a list of random values uniformly distributed in the list of closed intervals [(lo, hi)].
randomsInRangesM :: (RndM.RandomGen g, RndM.Random a) => V.Vector (a, a) -> RndM.Rand g (V.Vector a)
randomsInRangesM rangeList = mapM RndM.getRandomR rangeList
