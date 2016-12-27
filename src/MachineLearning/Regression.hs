{-|
Module: MachineLearning.Regression
Description: Regression
Copyright: (c) Alexander Ignatyev, 2016
License: BSD-3
Stability: experimental
Portability: POSIX

-}

module MachineLearning.Regression
(
  module Model
  , module LeastSquares
  , module Logistic
  , MinimizeMethod(..)
  , minimize
)

where

import Types
import MachineLearning.Regression.Model as Model
import MachineLearning.Regression.LeastSquares as LeastSquares
import MachineLearning.Regression.Logistic as Logistic
import qualified MachineLearning.Regression.GradientDescent as GD

import qualified Numeric.GSL.Minimization as Min

data MinimizeMethod = GradientDescent R       -- ^ Gradient descent, takes alpha. Requires feature normalization.
                    | ConjugateGradientFR R R -- ^ Fletcher-Reeves conjugate gradient algorithm,
                                              -- takes size of first trial step (0.1 is fine) and tol (0.1 is fine).
                    | ConjugateGradientPR R R -- ^ Polak-Ribiere conjugate gradient algorithm.
                                              -- takes size of first trial step (0.1 is fine) and tol (0.1 is fine).
                    | BFGS2 R R               -- ^ Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm,
                                              -- takes size of first trial step (0.1 is fine) and tol (0.1 is fine).


-- | Returns solution vector (theta) and optimization path.
-- Optimization path's row format:
-- [iter number, cost function value, theta values...]
minimize :: Model.Model a =>
            MinimizeMethod
         -> a       -- ^ model (Least Squares, Logistic Regression etc)
         -> R   -- ^ epsilon, desired precision of the solution
         -> Int     -- ^ maximum number of iterations allowed
         -> R   -- ^ regularization parameter lambda
         -> Matrix  -- ^ X
         -> Vector  -- ^ y
         -> Vector  -- ^ initial solution, theta
         -> (Vector, Matrix) -- ^ solution vector and optimization path

minimize (BFGS2 firstStepSize tol) = minimizeVD Min.VectorBFGS2 firstStepSize tol
minimize (ConjugateGradientFR firstStepSize tol) = minimizeVD Min.ConjugateFR firstStepSize tol
minimize (ConjugateGradientPR firstStepSize tol) = minimizeVD Min.ConjugatePR firstStepSize tol
minimize (GradientDescent alpha) = GD.gradientDescent alpha


minimizeVD method firstStepSize tol model epsilon niters lambda x y initialTheta
  = Min.minimizeVD method epsilon niters firstStepSize tol costf gradientf initialTheta
  where costf = Model.cost model lambda x y
        gradientf = Model.gradient model lambda x y
        
