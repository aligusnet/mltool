{-|
Module: MachineLearning.NeuralNetwork.MultiSvmLoss
Description: Multi SVM Loss.
Copyright: (c) Alexander Ignatyev, 2017
License: BSD-3
Stability: experimental
Portability: POSIX

Multi SVM Loss.
-}

module MachineLearning.NeuralNetwork.MultiSvmLoss
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


-- | SVM Delta
svmD :: R
svmD = 1.0


scores = id


gradient :: Matrix -> Matrix -> Matrix
gradient scores y =
    let nSamples = fromIntegral $ LA.rows scores
        correct_scores = sumByRows $ scores*(LA.step y)
        margins = scores - (correct_scores - (LA.scalar svmD))
        margins' = (1-y)*(LA.step margins)
        k = sumByRows margins'
    in margins' - (y * k)


loss :: Matrix -> Matrix -> R
loss scores y = 
  let nSamples = fromIntegral $ LA.rows scores
      correct_scores = sumByRows $ scores*(LA.step y)
      margins = scores - (correct_scores - (LA.scalar svmD))
      margins' = margins * (LA.step margins)
      loss = LA.sumElements margins' - nSamples * svmD
  in loss / nSamples
