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
  , Regularization(..)
  , ccostReg
  , cgradientReg
)

where

import MachineLearning.Types
import MachineLearning.Model
import MachineLearning.Regularization (Regularization(..))
import qualified MachineLearning as ML
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
  ccost :: a -> Regularization -> Matrix -> Vector -> Matrix -> R

  -- | Gradient function.
  -- It takes regularizarion parameter lambda, X (m x n), y (m x 1) and theta (n x 1).
  -- Returns vector of gradients (n x 1).
  cgradient :: a -> Regularization -> Matrix -> Vector -> Matrix -> Matrix

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


-- | Calculates regularization for Classifier.ccost.
-- It takes regularization parameter and theta.
ccostReg :: Regularization -> Matrix -> R
ccostReg RegNone _ = 0
ccostReg (L2 lambda) theta = (LA.norm_2 thetaReg) * 0.5 * lambda
  where thetaReg = ML.removeBiasDimension theta


-- | Calculates regularization for Classifier.cgradient.
-- It takes regularization parameter and theta.
cgradientReg :: Regularization -> Matrix -> Matrix
cgradientReg RegNone _ = 0
cgradientReg (L2 lambda) theta = ((LA.scalar lambda) * thetaReg)
  where thetaReg = 0 LA.||| (ML.removeBiasDimension theta)
