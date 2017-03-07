{-|
Module: MachineLearning.Optimization
Description: Optimization
Copyright: (c) Alexander Ignatyev, 2016
License: BSD-3
Stability: experimental
Portability: POSIX

Optimization module.
-}

module MachineLearning.Optimization
(
    MinimizeMethod(..)
  , minimize
  , checkGradient
)

where

import MachineLearning.Types (R, Vector, Matrix)
import MachineLearning.Model as Model
import MachineLearning.Regularization (Regularization)
import qualified MachineLearning.Optimization.GradientDescent as GD
import qualified MachineLearning.Optimization.MinibatchGradientDescent as MGD
import qualified Data.Vector.Storable as V
import qualified Numeric.LinearAlgebra as LA


import qualified Numeric.GSL.Minimization as Min

data MinimizeMethod = GradientDescent R       -- ^ Gradient descent, takes alpha. Requires feature normalization.
                    | MinibatchGradientDescent Int Int R  -- ^ Minibacth Gradietn Descent, takes seed, batch size and alpha
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
         -> Regularization   -- ^ regularization parameter
         -> Matrix  -- ^ X
         -> Vector  -- ^ y
         -> Vector  -- ^ initial solution, theta
         -> (Vector, Matrix) -- ^ solution vector and optimization path

minimize (BFGS2 firstStepSize tol) = minimizeVD Min.VectorBFGS2 firstStepSize tol
minimize (ConjugateGradientFR firstStepSize tol) = minimizeVD Min.ConjugateFR firstStepSize tol
minimize (ConjugateGradientPR firstStepSize tol) = minimizeVD Min.ConjugatePR firstStepSize tol
minimize (GradientDescent alpha) = GD.gradientDescent alpha
minimize (MinibatchGradientDescent seed batchSize alpha) = MGD.minibatchGradientDescent seed batchSize alpha


minimizeVD method firstStepSize tol model epsilon niters reg x y initialTheta
  = Min.minimizeVD method epsilon niters firstStepSize tol costf gradientf initialTheta
  where costf = Model.cost model reg x y
        gradientf = Model.gradient model reg x y

-- | Gradient checking function.
-- Approximates the derivates of the Model's cost function
-- and calculates derivatives using the Model's gradient functions.
-- Returns norn_2 between 2 derivatives.
-- Takes model, regularization, X, y, theta and epsilon (used to approximate derivatives, 1e-4 is a good value).
checkGradient :: Model a => a -> Regularization -> Matrix -> Vector -> Vector -> R -> R
checkGradient model reg x y theta eps = LA.norm_2 $ grad1 - grad2
  where theta_m = LA.asColumn theta
        eps_v = V.replicate (V.length theta) eps
        eps_m = LA.diag eps_v
        theta_m1 = theta_m + eps_m
        theta_m2 = theta_m - eps_m
        cost1 = LA.vector $ map (cost model reg x y) $ LA.toColumns theta_m1
        cost2 = LA.vector $ map (cost model reg x y) $ LA.toColumns theta_m2
        grad1 = (cost1 - cost2) / (LA.scalar $ 2*eps)
        grad2 = gradient model reg x y theta

