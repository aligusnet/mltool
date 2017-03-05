{-|
Module: MachineLearning.NeuralNetwork.SoftmaxLoss
Description: Softmax Loss.
Copyright: (c) Alexander Ignatyev, 2017
License: BSD-3
Stability: experimental
Portability: POSIX

Softmax Loss.
-}

module MachineLearning.NeuralNetwork.SoftmaxLoss
(
  scores
  , gradient
  , loss
)

where


import qualified Data.Vector.Storable as V
import qualified Numeric.LinearAlgebra as LA
import MachineLearning.Types (R, Matrix)
import MachineLearning.Utils (sumByRows, reduceByRows)

scores x = x - reduceByRows V.maximum x


gradient scores y =
  let sum_probs = sumByRows $ exp scores
      probs = (exp scores) / sum_probs
  in probs - y


-- Softmax Loss function
loss :: Matrix -> [(Matrix, Matrix)] -> Matrix -> R
loss scores thetaList y = LA.sumElements $ (log sum_probs) - t
  where m = fromIntegral $ LA.rows scores
        sum_probs = sumByRows $ exp scores
        t = sumByRows $ scores * y
