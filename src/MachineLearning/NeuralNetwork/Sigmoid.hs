{-|
Module: MachineLearning.NeuralNetwork.Sigmoid
Description: Sigmoid
Copyright: (c) Alexander Ignatyev, 2017
License: BSD-3
Stability: experimental
Portability: POSIX

Sigmoid
-}

module MachineLearning.NeuralNetwork.Sigmoid
(
    LM.sigmoid
    , gradient
    , loss
    , outputGradient
)

where


import qualified Numeric.LinearAlgebra as LA
import MachineLearning.Types (R, Matrix)
import qualified MachineLearning.LogisticModel as LM


gradient :: Matrix -> Matrix -> Matrix
gradient z da = da * LM.sigmoidGradient z


outputGradient :: Matrix -> Matrix -> Matrix
outputGradient scores y = scores - y


-- Sigmoid Loss function
loss :: Matrix -> Matrix -> R
loss x y = (LA.sumElements $ (-y) * log(tau + x) - (1-y) * log ((1+tau)-x))/m
  where tau = 1e-7
        m = fromIntegral $ LA.rows x
