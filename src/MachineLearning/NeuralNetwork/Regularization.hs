{-|
Module: MachineLearning.NeuralNetwork.Regularization
Description: Regularization
Copyright: (c) Alexander Ignatyev, 2017
License: BSD-3
Stability: experimental
Portability: POSIX

Regularization.
-}

module MachineLearning.NeuralNetwork.Regularization
(
  Regularization(..)
  , forwardReg
  , backwardReg
)

where

import MachineLearning.Types (R, Matrix)
import qualified Numeric.LinearAlgebra as LA


data Regularization = L2 Double


forwardReg :: Regularization -> [(Matrix, Matrix)] -> R
forwardReg (L2 lambda) thetaList = 0.5 * lambda * (sum $ map LA.norm_2 $ snd $ unzip thetaList)


backwardReg :: Regularization -> Matrix -> Matrix
backwardReg (L2 lambda) w = w * (LA.scalar lambda)
