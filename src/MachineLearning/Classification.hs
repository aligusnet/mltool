{-|
Module: MachineLearning.Classification
Description: Classification.
Copyright: (c) Alexander Ignatyev, 2016
License: BSD-3
Stability: experimental
Portability: POSIX

Binary and Multiclass Classification.
-}

module MachineLearning.Classification
(
  predictBinary
  , predictMulti
  , calcAccuracy
  , processOutputMulti
  , learnBinary
  , learnMulti
)

where

import Types
import qualified MachineLearning.Regression as MLR
import qualified Data.Vector.Storable as V
import qualified Numeric.LinearAlgebra as LA


-- | Binary Classification prediction function.
-- Takes a matrix of features X and a vector theta.
-- Returns predicted class.
predictBinary :: Matrix -> Vector -> Vector
predictBinary x theta = V.map (\r -> if r >= 0.5 then 1 else 0) $ predict x theta


-- | Multiclass Classification prediction function.
-- Takes a matrix of features X and a list of vectors theta,
-- returns predicted class number assuming that class numbers start at 0.
predictMulti :: Matrix -> [Vector] -> Vector
predictMulti x thetas = predictions'
  where predictions = LA.toRows . LA.fromColumns $ map (predict x) thetas
        predictions' = LA.vector $ map (fromIntegral . LA.maxIndex) predictions


-- | Binary Classification prediction fucntion.
-- Takes a matrix of features X and a vector theta.
-- Returns probability of positive class.
predict :: Matrix -> Vector -> Vector
predict x theta = MLR.hypothesis MLR.Logistic x theta


-- | Calculates accuracy of Classification predictions.
-- Takes matrix X, vector expected y and vector predicted y.
-- Returns number from 0 to 1, the closer to 1 the better accuracy.
-- Suitable for both Classification Types: Binary and Multiclass.
calcAccuracy :: Matrix -> Vector -> Vector -> R
calcAccuracy x yExpected yPredicted = (1 - (V.sum discrepancy) / (fromIntegral $ V.length discrepancy))
  where discrepancy = V.zipWith f yExpected yPredicted
        f y1 y2 = if round y1 == round y2 then 0 else 1


-- | Process outputs for Multiclass Classification.
-- Takes number of labels and output vector y.
-- Returns list of vectors of binary outputs (One-vs-All Classification).
-- It is supposed that labels are integerets start at 0.
processOutputMulti :: Int -> Vector -> [Vector]
processOutputMulti numLabels y = map f [0 .. numLabels-1]
  where f sample = V.map (\a -> if round a == sample then 1 else 0) y


-- | Learns Binary Classification.
learnBinary :: R                 -- ^ epsilon, desired precision of the solution;
            -> Int               -- ^ maximum number of iterations allowed;
            -> R                 -- ^ regularization parameter lambda;
            -> Matrix            -- ^ matrix X;
            -> Vector            -- ^ binary vector y;
            -> Vector            -- ^ initial Theta;
            -> (Vector, Matrix)  -- ^ solution vector and optimization path.
learnBinary eps numIters lambda x y initialTheta = MLR.minimize (MLR.BFGS2 0.1 0.1) MLR.Logistic eps numIters lambda x y initialTheta


-- | Learns Multiclass Classification
learnMulti :: R                 -- ^ epsilon, desired precision of the solution;
           -> Int               -- ^ maximum number of iterations allowed;
           -> R                 -- ^ regularization parameter lambda;
           -> Matrix            -- ^ matrix X;
           -> [Vector]          -- ^ list binary vector's y (one for every class, see `processOutputMulti` function);
           -> Vector            -- ^ initial theta;
           -> ([Vector], [Matrix])  -- ^ solution vector and optimization path.
learnMulti eps numIters lambda x ys initialTheta =
  let minimize y = MLR.minimize (MLR.BFGS2 0.1 0.1) MLR.Logistic eps numIters lambda x y initialTheta
  in unzip $ map minimize ys
