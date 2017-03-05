{-|
Module: MachineLearning.Utils
Description: Utils
Copyright: (c) Alexander Ignatyev, 2016
License: BSD-3
Stability: experimental
Portability: POSIX

Various helpful utilities.
-}

module MachineLearning.Utils
(
  reduceByRowsV
  , reduceByColumnsV
  , reduceByRows
  , reduceByColumns
  , sumByRows
  , sumByColumns
  , listOfTuplesToList
)

where

  
import MachineLearning.Types (R, Vector, Matrix)
import qualified Data.Vector.Storable as V
import qualified Numeric.LinearAlgebra as LA


reduceByRowsV :: (Vector -> R) -> Matrix -> Vector
reduceByRowsV f = LA.vector . map f . LA.toRows


reduceByColumnsV :: (Vector -> R) -> Matrix -> Vector
reduceByColumnsV f = LA.vector . map f . LA.toColumns


reduceByRows :: (Vector -> R) -> Matrix -> Matrix
reduceByRows f = LA.asColumn . reduceByRowsV f


reduceByColumns :: (Vector -> R) -> Matrix -> Matrix
reduceByColumns f = LA.asRow . reduceByColumnsV f


sumByColumns :: Matrix -> Matrix
sumByColumns = reduceByColumns V.sum


sumByRows :: Matrix -> Matrix
sumByRows = reduceByRows V.sum


-- | Converts list of tuples into list.
listOfTuplesToList :: [(a, a)] -> [a]
listOfTuplesToList [] = []
listOfTuplesToList ((a, b):xs) = a : b : listOfTuplesToList xs
