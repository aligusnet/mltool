{-|
Module: MachineLearning.SoftmaxClassifier
Description: Softmax Classifier.
Copyright: (c) Alexander Ignatyev, 2017-2018.
License: BSD-3
Stability: experimental
Portability: POSIX

Softmax Classifier (Multiclass Logistic Regression).
-}

module MachineLearning.SoftmaxClassifier
(
  module MachineLearning.Model
  , module MachineLearning.Classification.MultiClass
  , SoftmaxClassifier(..)
)

where

import Prelude hiding ((<>))
import qualified Numeric.LinearAlgebra as LA
import Numeric.LinearAlgebra((<>), (<.>), (|||))
import qualified Data.Vector.Storable as V

import qualified MachineLearning as ML
import MachineLearning.Types (R, Vector, Matrix)
import MachineLearning.Utils (reduceByRows, sumByRows)
import MachineLearning.Model
import MachineLearning.Classification.MultiClass


-- | Softmax Classifier, takes number of classes.
data SoftmaxClassifier = Softmax Int

instance Classifier SoftmaxClassifier where
  cscore (Softmax _) x theta = scores - reduceByRows V.maximum scores
    where scores = x <> (LA.tr theta)

  chypothesis m x theta = V.fromList predictions
    where scores = cscore m x theta
          scores' = LA.toRows scores
          predictions = map (fromIntegral . LA.maxIndex) scores'

  ccost m lambda x y theta =
    let nSamples = fromIntegral $ LA.rows x
        scores = cscore m x theta
        sum_probs = sumByRows $ exp scores
        loss = LA.sumElements $ (log sum_probs) - remap scores y
        reg = ccostReg lambda theta
    in (loss + reg) / nSamples 

  cgradient m lambda x y theta =
    let nSamples = fromIntegral $ LA.rows x
        ys = processOutput m y
        scores = cscore m x theta
        sum_probs = sumByRows $ exp scores
        probs = (exp scores) / sum_probs
        probs' = probs - ys
        dw =  (LA.tr probs') <> x
        reg = cgradientReg lambda theta
    in (dw + reg)/ nSamples

  cnumClasses (Softmax nLabels) = nLabels


remap :: Matrix -> Vector -> Matrix
remap m v = LA.remap cols rows m
  where cols = LA.asColumn $ V.fromList [0..(fromIntegral $ LA.rows m)-1]
        rows = LA.toInt $ LA.asColumn v

