{-|
Module: MachineLearning.Regression
Description: Regression
Copyright: (c) Alexander Ignatyev, 2016-2018.
License: BSD-3
Stability: experimental
Portability: POSIX
-}

module MachineLearning.Regression
(
  Model.Model(..)
  , LeastSquares.LeastSquaresModel(..)
  , Optimization.MinimizeMethod(..)
  , Optimization.minimize
  , normalEquation
  , normalEquation_p
  , Regularization(..)
)

where

import Prelude hiding ((<>))
import MachineLearning.Types (Vector, Matrix)
import MachineLearning.Optimization as Optimization
import MachineLearning.Model as Model
import MachineLearning.LeastSquaresModel as LeastSquares
import MachineLearning.Regularization (Regularization(..))

import qualified Numeric.LinearAlgebra as LA
import Numeric.LinearAlgebra ((<>), (#>))


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
