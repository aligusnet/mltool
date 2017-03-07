{-|
Module: MachineLearning.NeuralNetwork.SigmoidActivation
Description: Sigmoid
Copyright: (c) Alexander Ignatyev, 2017
License: BSD-3
Stability: experimental
Portability: POSIX

Sigmoid Activation.
-}

module MachineLearning.NeuralNetwork.SigmoidActivation
(
    LM.sigmoid
    , gradient
)

where


import MachineLearning.Types (R, Matrix)
import qualified MachineLearning.LogisticModel as LM


gradient :: Matrix -> Matrix -> Matrix
gradient z da = da * LM.sigmoidGradient z
