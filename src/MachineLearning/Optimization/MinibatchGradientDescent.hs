{-|
Module: MachineLearning.Optimization.MinibatchGradientDescent
Description: Gradient Descent
Copyright: (c) Alexander Ignatyev, 2017
License: BSD-3
Stability: experimental
Portability: POSIX

Minibatch Gradient Descent
-}

module MachineLearning.Optimization.MinibatchGradientDescent
(
  minibatchGradientDescent
)

where

import MachineLearning.Types (R, Vector, Matrix)
import qualified Data.Vector.Storable as V
import qualified Numeric.LinearAlgebra as LA
import Numeric.LinearAlgebra ((?))
import qualified Control.Monad.Random as RndM

import qualified MachineLearning.Model as Model

-- | Minibatch Gradient Descent method implementation. See "MachineLearning.Regression" for usage details.
minibatchGradientDescent :: Model.Model a
                            => Int              -- ^ seed
                            -> Int              -- ^ batch size
                            -> R                -- ^ learning rate, alpha
                            -> a                -- ^ model to learn
                            -> R                -- ^ epsilon
                            -> Int              -- ^ max number of iters
                            -> R                -- ^ regularization parameter, lambda
                            -> Matrix           -- ^ matrix of features, X
                            -> Vector           -- ^ output vector, y
                            -> Vector           -- ^ vector of initial weights, theta or w
                            -> (Vector, Matrix) -- ^ vector of weights and learning path 
minibatchGradientDescent seed batchSize alpha model eps maxIters lambda x y theta =
  RndM.evalRand (minibatchGradientDescentM batchSize alpha model eps maxIters lambda x y theta) gen
  where gen = RndM.mkStdGen seed


-- | Minibatch Gradient Descent method implementation. See "MachineLearning.Regression" for usage details.
minibatchGradientDescentM :: (Model.Model a, RndM.RandomGen g)
                             => Int              -- ^ batch size
                             -> R                -- ^ learning rate, alpha
                             -> a                -- ^ model to learn
                             -> R                -- ^ epsilon
                             -> Int              -- ^ max number of iters
                             -> R                -- ^ regularization parameter, lambda
                             -> Matrix           -- ^ matrix of features, X
                             -> Vector           -- ^ output vector, y
                             -> Vector           -- ^ vector of initial weights, theta or w
                             -> RndM.Rand g (Vector, Matrix) -- ^ vector of weights and learning path 
minibatchGradientDescentM batchSize alpha model eps maxIters lambda x y theta = do
  idxList <- RndM.getRandomRs (0, (LA.rows x) - 1)
  let gradient = Model.gradient model lambda
      cost = Model.cost model lambda
      helper theta nIters optPath =
          let idx = take batchSize idxList
              x' = x ? idx
              y' = LA.flatten $ (LA.asColumn y) ? idx
              theta' = theta - (LA.scale alpha (gradient x' y' theta))
              j = cost x' y' theta'
              gradientTest = LA.norm_2 (theta' - theta) < eps
              optPathRow = V.concat [LA.vector [(fromIntegral $ maxIters - nIters), j], theta']
              optPath' = optPathRow : optPath
          in if gradientTest || nIters <= 1
             then (theta, LA.fromRows $ reverse optPath')
             else helper theta' (nIters - 1) optPath'
  return $ helper theta maxIters []

