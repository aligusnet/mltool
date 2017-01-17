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
    Opt.MinimizeMethod(..)
  , module Log
  , module Model
  , predictBinary
  , predictMulti
  , calcAccuracy
  , processOutputMulti
  , learnBinary
  , learnMulti
)

where

import MachineLearning.Types (R, Vector, Matrix)
import qualified MachineLearning.Optimization as Opt
import qualified MachineLearning.LogisticModel as Log
import qualified MachineLearning.Model as Model
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
predict x theta = Model.hypothesis Log.Logistic x theta


-- | Calculates accuracy of Classification predictions.
-- Takes vector expected y and vector predicted y.
-- Returns number from 0 to 1, the closer to 1 the better accuracy.
-- Suitable for both Classification Types: Binary and Multiclass.
calcAccuracy :: Vector -> Vector -> R
calcAccuracy yExpected yPredicted = (1 - (V.sum discrepancy) / (fromIntegral $ V.length discrepancy))
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
learnBinary :: Opt.MinimizeMethod -- ^ (e.g. BFGS2 0.1 0.1)
            -> R                  -- ^ epsilon, desired precision of the solution;
            -> Int                -- ^ maximum number of iterations allowed;
            -> R                  -- ^ regularization parameter lambda;
            -> Matrix             -- ^ matrix X;
            -> Vector             -- ^ binary vector y;
            -> Vector             -- ^ initial Theta;
            -> (Vector, Matrix)   -- ^ solution vector and optimization path.
learnBinary mm eps numIters lambda x y initialTheta = Opt.minimize mm Log.Logistic eps numIters lambda x y initialTheta


-- | Learns Multiclass Classification
learnMulti :: Opt.MinimizeMethod -- ^ (e.g. BFGS2 0.1 0.1)
           -> R                  -- ^ epsilon, desired precision of the solution;
           -> Int                -- ^ maximum number of iterations allowed;
           -> R                  -- ^ regularization parameter lambda;
           -> Matrix             -- ^ matrix X;
           -> [Vector]           -- ^ list binary vector's y (one for every class, see `processOutputMulti` function);
           -> [Vector]           -- ^ initial theta;
           -> ([Vector], [Matrix])  -- ^ solution vector and optimization path.
learnMulti mm eps numIters lambda x ys initialThetaList =
  let minimize = Opt.minimize mm Log.Logistic eps numIters lambda x
  in unzip $ zipWith minimize ys initialThetaList
