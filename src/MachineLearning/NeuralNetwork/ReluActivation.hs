{-|
Module: MachineLearning.NeuralNetwork.ReluActivation
Description: ReLu Activation
Copyright: (c) Alexander Ignatyev, 2017
License: BSD-3
Stability: experimental
Portability: POSIX

ReLu Activation.
-}

module MachineLearning.NeuralNetwork.ReluActivation
(
  relu
  , gradient
)

where


import qualified Numeric.LinearAlgebra as LA
import MachineLearning.Types (Matrix)


relu :: Matrix -> Matrix
relu x = x * (LA.step x)


gradient :: Matrix -> Matrix -> Matrix
gradient x dx = dx * (LA.step x)  -- == dx[x<0] = 0
