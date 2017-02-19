{-|
Module: MachineLearning.Classification.MultiClass
Description: MultiClass Classification.
Copyright: (c) Alexander Ignatyev, 2016-2017
License: BSD-3
Stability: experimental
Portability: POSIX

MultiClass Classification.
-}

module MachineLearning.Classification.MultiClass
(
  Classifier(..)
  , MultiClassModel(..)
  , processOutput
)

where

import MachineLearning.Types
import MachineLearning.Model
import qualified Data.Vector.Storable as V
import qualified Numeric.LinearAlgebra as LA


-- | Classifier type class represents Multi-class classification models.
class Classifier a where
  -- | Score function
  cscore :: a -> Matrix -> Matrix -> Matrix

  -- | Hypothesis function
  -- Takes X (m x n) and theta (n x k), returns y (m x k).
  chypothesis :: a -> Matrix -> Matrix -> Vector
  
  -- | Cost function J(Theta), a.k.a. loss function.
  -- It takes regularizarion parameter lambda, matrix X (m x n), vector y (m x 1) and vector theta (n x 1).
  ccost :: a -> R -> Matrix -> Vector -> Matrix -> R

  -- | Gradient function.
  -- It takes regularizarion parameter lambda, X (m x n), y (m x 1) and theta (n x 1).
  -- Returns vector of gradients (n x 1).
  cgradient :: a -> R -> Matrix -> Vector -> Matrix -> Matrix

  -- | Returns Number of Classes
  cnumClasses :: a -> Int


-- | MultiClassModel is Model wrapper class around Classifier
data MultiClassModel m = MultiClass m


instance (Classifier a) => Model (MultiClassModel a) where
  hypothesis (MultiClass m) x theta = chypothesis m x theta'
    where theta' = unflatten (cnumClasses m) theta

  cost (MultiClass m) lambda x y theta = ccost m lambda x y theta'
    where theta' = unflatten (cnumClasses m) theta

  gradient (MultiClass m) lambda x y theta = LA.flatten $ cgradient m lambda x y theta'
    where theta' = unflatten (cnumClasses m) theta


unflatten :: Int -> Vector -> Matrix
unflatten nLabels v = LA.reshape cols v
  where cols = (V.length v) `div` nLabels


-- | Process outputs for MultiClass Classification.
-- Takes Classifier and output vector y.
-- Returns matrix of binary outputs.
-- It is supposed that labels are integerets start at 0.
processOutput :: (Classifier c) => c -> Vector -> Matrix
processOutput c y = LA.fromColumns $ map f [0 .. (cnumClasses c)-1]
  where f sample = V.map (\a -> if round a == sample then 1 else 0) y
