{-|
Module: MachineLearning
Description: Machine Learning
Copyright: (c) Alexander Ignatyev, 2016
License: BSD-3
Stability: experimental
Portability: POSIX

-}


module MachineLearning
(
  normalEquation
  , normalEquation_p
  , addColumnOfOnes
  , meanStddev
  , featureNormalization
  , mapFeatures
  , splitToXY
)

where

import Types
import qualified Numeric.LinearAlgebra as LA
import Numeric.LinearAlgebra((<>), (#>), (|||), (??))
import qualified Numeric.GSL.Statistics as Stat

import Control.Monad (replicateM, mfilter, MonadPlus)
import Data.List (sort, group, foldl')
import qualified Data.Vector as V


-- | Normal equation using inverse, does not require feature normalization
-- It takes X and y, returns theta.
normalEquation :: Matrix -> Vector -> Vector
normalEquation x y =
  let trX = LA.tr x
  in (LA.inv (trX <> x) <> trX) #> y


-- | Normal equation using pseudo inverse, requires feature normalization
-- It takes X and y, returns theta.
normalEquation_p :: Matrix -> Vector -> Vector
normalEquation_p x y =
  let trX = LA.tr x
  in (LA.pinv (trX <> x) <> trX) #> y


-- | Add column of ones to feature matrix.
addColumnOfOnes :: Matrix -> Matrix
addColumnOfOnes x = 1 ||| x


-- | Caclulates mean and stddev values of every feature.
-- Takes feature matrix X, returns pair of vectors of means and stddevs.
meanStddev x =
  let cols = LA.toColumns x
      means = map Stat.mean cols
      stddevs = zipWith (\m col -> Stat.stddev_m m col) means cols
      stddevs' = map (\s -> if s < 2 then 1 else s) stddevs
  in (LA.row means, LA.row stddevs')


featureNormalization (means, stddevs) x = (x - means) / stddevs

-- | Maps the features into all polynomial terms of X up to the degree-th power
mapFeatures :: Int -> Matrix -> Matrix
mapFeatures 1 x = x
mapFeatures degree x = LA.fromColumns $ cols ++ (foldl' (\l d -> (terms d) ++ l) [] [degree, degree-1 .. 2])
  where cols = LA.toColumns x
        vv = V.fromList cols
        ncols = V.length vv
        makeTerm :: [(Int, Int)] -> Vector
        makeTerm = foldl' (\c (index, power) -> c * (vv V.! index) ^ power) 1
        terms :: Int -> [Vector]
        terms d = foldl' (\l x -> (makeTerm x) : l) [] $ polynomialTerms d [ncols-1, ncols-2 .. 0]
        

polynomialTerms :: Ord a => Int -> [a] -> [[(a, Int)]]
polynomialTerms degree terms =
  map (\x -> map (\y -> (head y, length y)) $ group x)
  $ combinationsWithReplacement degree terms


combinationsWithReplacement :: (MonadPlus m, Ord a) => Int -> m a -> m [a]
combinationsWithReplacement sample objects = mfilter (\a -> sort a == a) $ replicateM sample objects


-- | Splits data matrix to features matrix X and vector of outputs y
splitToXY m =
  let x = m ?? (LA.All, LA.DropLast 1)
      y = LA.flatten $ m ?? (LA.All, LA.TakeLast 1)
  in (x, y)
