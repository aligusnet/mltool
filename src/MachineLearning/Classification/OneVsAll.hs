{-|
Module: MachineLearning.Classification.OneVsAll
Description: One-vs-All Classification.
Copyright: (c) Alexander Ignatyev, 2016-2017
License: BSD-3
Stability: experimental
Portability: POSIX

One-vs-All Classification.
-}

module MachineLearning.Classification.OneVsAll
(
    Opt.MinimizeMethod(..)
  , module Log
  , module Model
  , predict
  , learn
  , MLC.calcAccuracy
  , Regularization(..)
)

where

import MachineLearning.Types (R, Vector, Matrix)
import MachineLearning.Regularization (Regularization(..))
import qualified MachineLearning.Optimization as Opt
import qualified MachineLearning.LogisticModel as Log
import qualified MachineLearning.Model as Model
import qualified MachineLearning.Classification.Internal as MLC
import qualified Data.Vector.Storable as V
import qualified Numeric.LinearAlgebra as LA


-- | One-vs-All Classification prediction function.
-- Takes a matrix of features X and a list of vectors theta,
-- returns predicted class number assuming that class numbers start at 0.
predict :: Matrix -> [Vector] -> Vector
predict x thetas = predictions'
  where predict = Model.hypothesis Log.Logistic x
        predictions = LA.toRows . LA.fromColumns $ map predict thetas
        predictions' = LA.vector $ map (fromIntegral . LA.maxIndex) predictions


-- | Learns One-vs-All Classification
learn :: Opt.MinimizeMethod -- ^ (e.g. BFGS2 0.1 0.1)
         -> R                  -- ^ epsilon, desired precision of the solution;
         -> Int                -- ^ maximum number of iterations allowed;
         -> Regularization     -- ^ regularization parameter lambda;
         -> Int                -- ^ number of labels
         -> Matrix             -- ^ matrix X;
         -> Vector             -- ^ vector y
         -> [Vector]             -- ^ initial theta list;
         -> ([Vector], [Matrix])  -- ^ solution vector and optimization path.
learn mm eps numIters lambda nLabels x y initialThetaList =
  let ys = MLC.processOutputOneVsAll nLabels y
      minimize = Opt.minimize mm Log.Logistic eps numIters lambda x
  in unzip $ zipWith minimize ys initialThetaList
