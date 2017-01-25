{-|
Module: MachineLearning.Classification.Internal
Description: Classification Internal module.
Copyright: (c) Alexander Ignatyev, 2016-2017
License: BSD-3
Stability: experimental
Portability: POSIX

Defines Internal Classification functions.
-}

module MachineLearning.Classification.Internal
(
  calcAccuracy
  , processOutputOneVsAll
)

where

import MachineLearning.Types (R, Vector)
import qualified Data.Vector.Storable as V


-- | Calculates accuracy of Classification predictions.
-- Takes vector expected y and vector predicted y.
-- Returns number from 0 to 1, the closer to 1 the better accuracy.
-- Suitable for both Classification Types: Binary and Multiclass.
calcAccuracy :: Vector -> Vector -> R
calcAccuracy yExpected yPredicted = (1 - (V.sum discrepancy) / (fromIntegral $ V.length discrepancy))
  where discrepancy = V.zipWith f yExpected yPredicted
        f y1 y2 = if round y1 == round y2 then 0 else 1


-- | Process outputs for One-vs-All Classification.
-- Takes number of labels and output vector y.
-- Returns list of vectors of binary outputs (One-vs-All Classification).
-- It is supposed that labels are integerets start at 0.
processOutputOneVsAll :: Int -> Vector -> [Vector]
processOutputOneVsAll numLabels y = map f [0 .. numLabels-1]
  where f sample = V.map (\a -> if round a == sample then 1 else 0) y
