{-|
Module: MachineLearning.NeuralNetwork.TanhActivation.
Description: Tanh Activation.
Copyright: (c) Alexander Ignatyev, 2017
License: BSD-3
Stability: experimental
Portability: POSIX

Tanh Activation.
-}

module MachineLearning.NeuralNetwork.TanhActivation
(
  tanh
  , gradient
)

where


import qualified Numeric.LinearAlgebra as LA
import MachineLearning.Types (Matrix)


tanhGradient :: Matrix -> Matrix
tanhGradient x = 1 - tanhx*tanhx
  where tanhx = tanh x


gradient :: Matrix -> Matrix -> Matrix
gradient x dx = dx * (tanhGradient x)

