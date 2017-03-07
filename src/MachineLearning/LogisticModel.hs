{-|
Module: MachineLearning.LogisticModel
Description: Logistic Regression Model
Copyright: (c) Alexander Ignatyev, 2016
License: BSD-3
Stability: experimental
Portability: POSIX

-}

module MachineLearning.LogisticModel
(
  module MachineLearning.Model
  , LogisticModel(..)
  , sigmoid
  , sigmoidGradient
)

where

import qualified Numeric.LinearAlgebra as LA
import Numeric.LinearAlgebra((<>), (#>), (<.>))
import qualified Data.Vector.Storable as V

import MachineLearning.Model
import qualified MachineLearning.Regularization as R

data LogisticModel = Logistic


-- | Calculates sigmoid
sigmoid :: Floating a => a -> a
sigmoid z = 1 / (1+exp(-z))


-- | Calculates derivatives of sigmoid
sigmoidGradient :: Floating a => a -> a
sigmoidGradient z = s * (1-s)
  where s = sigmoid z


instance Model LogisticModel where
  hypothesis Logistic x theta = sigmoid (x #> theta)

  cost m lambda x y theta =
    let h = hypothesis m x theta
        nFeatures = V.length theta
        nExamples = fromIntegral $ LA.rows x
        tau = 1e-7
        jPositive = log(tau + h) <.> (-y)
        jNegative = log((1 + tau) - h) <.> (1-y)
        regTerm = R.costReg lambda theta
    in (jPositive - jNegative + regTerm) / nExamples

  gradient m lambda x y theta = (((LA.tr x) #> (h  - y)) + regTerm) / nExamples
    where h = hypothesis m x theta
          nExamples = fromIntegral $ LA.rows x
          regTerm = R.gradientReg lambda theta
