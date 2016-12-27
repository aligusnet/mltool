{-|
Module: MachineLearning.Regression.GradientDescent
Description: Gradient Descent
Copyright: (c) Alexander Ignatyev, 2016
License: BSD-3
Stability: experimental
Portability: POSIX

-}

module MachineLearning.Regression.GradientDescent
(
  gradientDescent
)

where

import Types
import qualified Data.Vector.Storable as V
import qualified Numeric.LinearAlgebra as LA

import qualified MachineLearning.Regression.Model as Model

-- | Gradient Descent method implementation. See "MachineLearning.Regression" for usage details.
gradientDescent :: Model.Model a => R-> a -> R -> Int -> R -> Matrix -> Vector -> Vector -> (Vector, Matrix)
gradientDescent alpha model eps maxIters lambda x y theta = helper theta maxIters []
  where gradient = Model.gradient model lambda
        cost = Model.cost model lambda
        helper theta nIters optPath =
          let theta' = theta - (LA.scale alpha (gradient x y theta))
              j = cost x y theta'
              gradientTest = LA.norm_2 (theta' - theta) < eps
              optPathRow = V.concat [LA.vector [(fromIntegral $ maxIters - nIters), j], theta']
              optPath' = optPathRow : optPath
          in if gradientTest || nIters <= 1
             then (theta, LA.fromRows $ reverse optPath')
             else helper theta' (nIters - 1) optPath'
