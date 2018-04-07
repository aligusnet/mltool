{-|
Module: MachineLearning.MultiSvmClassifier
Description: Multiclass Support Vector Machines Classifier.
Copyright: (c) Alexander Ignatyev, 2017-2018.
License: BSD-3
Stability: experimental
Portability: POSIX

Multicalss Support Vector Machines Classifier.
-}

module MachineLearning.MultiSvmClassifier
(
  module MachineLearning.Model
  , module MachineLearning.Classification.MultiClass
  , MultiSvmClassifier(..)
)

where


import Prelude hiding ((<>))
import qualified Numeric.LinearAlgebra as LA
import Numeric.LinearAlgebra((<>), (<.>), (|||))
import qualified Data.Vector.Storable as V

import qualified MachineLearning as ML
import MachineLearning.Types (R, Vector, Matrix)
import MachineLearning.Utils (sumByRows, reduceByRowsV)
import MachineLearning.Model
import MachineLearning.Classification.MultiClass


-- | Multiclass SVM Classifier, takes delta and number of futures. Delta = 1.0 is good for all cases.
data MultiSvmClassifier = MultiSvm R Int


instance Classifier MultiSvmClassifier where
  cscore (MultiSvm _ _) x theta = x <> (LA.tr theta)

  chypothesis m x theta = predictions
    where scores = cscore m x theta
          predictions = reduceByRowsV (fromIntegral . LA.maxIndex) scores

  ccost m@(MultiSvm d _) lambda x y theta =
    let nSamples = fromIntegral $ LA.rows x
        scores = cscore m x theta
        correct_scores = LA.remap (LA.asColumn $ V.fromList [0..(fromIntegral $ LA.rows x)-1]) (LA.toInt $ LA.asColumn y) scores
        margins = scores - (correct_scores - (LA.scalar d))
        margins' = margins * LA.step margins
        loss = LA.sumElements(margins') / nSamples - d
        reg = (ccostReg lambda theta) / nSamples
    in loss + reg

  cgradient m@(MultiSvm d _) lambda x y theta =
    let nSamples = fromIntegral $ LA.rows x
        ys = processOutput m y
        scores = cscore m x theta
        correct_scores = LA.remap (LA.asColumn $ V.fromList [0..(fromIntegral $ LA.rows x)-1]) (LA.toInt $ LA.asColumn y) scores
        margins = scores - (correct_scores - (LA.scalar d))
        margins' = (1-ys)*(LA.step margins)  -- step == cmap (\x -> if x>0 then 1 else 0)
        k = sumByRows margins'
        margins'' = margins' - (ys * k)
        dw = ((LA.tr margins'') <> x) / nSamples
        reg = (cgradientReg lambda theta) / nSamples
     in dw + reg

  cnumClasses (MultiSvm _ nLabels) = nLabels
