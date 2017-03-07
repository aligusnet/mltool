{-|
Module: MachineLearning.NeuralNetwork.LogisticLoss
Description: Multi SVM Loss.
Copyright: (c) Alexander Ignatyev, 2017
License: BSD-3
Stability: experimental
Portability: POSIX

Logistic Loss.
-}

module MachineLearning.NeuralNetwork.LogisticLoss
(
  scores
  , gradient
  , loss
)

where


import qualified Numeric.LinearAlgebra as LA
import MachineLearning.Types (R, Matrix)
import qualified MachineLearning.LogisticModel as LM


scores :: Matrix -> Matrix
scores = LM.sigmoid


gradient :: Matrix -> Matrix -> Matrix
gradient scores y = scores - y


-- Logistic Loss function
loss :: Matrix -> Matrix -> R
loss x y = (LA.sumElements $ (-y) * log(tau + x) - (1-y) * log ((1+tau)-x))/m
  where tau = 1e-7
        m = fromIntegral $ LA.rows x
