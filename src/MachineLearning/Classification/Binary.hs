{-|
Module: MachineLearning.Classification.Binary
Description: Binary Classification.
Copyright: (c) Alexander Ignatyev, 2016-2017
License: BSD-3
Stability: experimental
Portability: POSIX

Binary Classification.
-}

module MachineLearning.Classification.Binary
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


-- | Binary Classification prediction function.
-- Takes a matrix of features X and a vector theta.
-- Returns predicted class.
predict :: Matrix -> Vector -> Vector
predict x theta = V.map (\r -> if r >= 0.5 then 1 else 0) h
  where h = Model.hypothesis Log.Logistic x theta


-- | Learns Binary Classification.
learn :: Opt.MinimizeMethod -- ^ (e.g. BFGS2 0.1 0.1)
         -> R                  -- ^ epsilon, desired precision of the solution;
         -> Int                -- ^ maximum number of iterations allowed;
         -> Regularization     -- ^ regularization parameter lambda;
         -> Matrix             -- ^ matrix X;
         -> Vector             -- ^ binary vector y;
         -> Vector             -- ^ initial Theta;
         -> (Vector, Matrix)   -- ^ solution vector and optimization path.
learn mm eps numIters lambda x y initialTheta = Opt.minimize mm Log.Logistic eps numIters lambda x y initialTheta
