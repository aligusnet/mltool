{-|
Module: MachineLearning.Regression
Description: Regression
Copyright: (c) Alexander Ignatyev, 2016-2017
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
)

where

import MachineLearning.Optimization as Optimization
import MachineLearning.Model as Model
import MachineLearning.LeastSquaresModel as LeastSquares
