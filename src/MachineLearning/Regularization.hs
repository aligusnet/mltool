{-|
Module: MachineLearning.Regularization
Description: Regularization
Copyright: (c) Alexander Ignatyev, 2017
License: BSD-3
Stability: experimental
Portability: POSIX

Regularization.
-}

module MachineLearning.Regularization
(
  Regularization(..)
  , costReg
  , gradientReg
)

where

import MachineLearning.Types (R, Vector)
import qualified Data.Vector.Storable as V
import qualified Numeric.LinearAlgebra as LA

-- | Regularization
data Regularization = RegNone -- ^ No regularization
                    | L2 R    -- ^ L2 Regularization



-- | Calculates regularization for Model.cost function.
-- It takes regularization parameter and theta.
costReg :: Regularization -> Vector -> R
costReg RegNone _ = 0
costReg (L2 lambda) theta = (thetaReg LA.<.> thetaReg) * lambda * 0.5
  where thetaReg = V.tail theta



-- | Calculates regularization for Model.gradient function.
-- It takes regularization parameter and theta.
gradientReg :: Regularization -> Vector -> Vector
gradientReg RegNone _ = 0
gradientReg (L2 lambda) theta = (LA.scalar lambda) * thetaReg
  where thetaReg = theta V.// [(0, 0)]
