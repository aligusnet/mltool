{-|
Module: MachineLearning.SoftmaxModel
Description: Softmax Model.
Copyright: (c) Alexander Ignatyev, 2017.
License: BSD-3
Stability: experimental
Portability: POSIX

Softmax Model (Multiclass Logistic Regression).
-}

module MachineLearning.SoftmaxModel
(
  module MachineLearning.Model
  , SoftmaxModel(..)
)

where

import qualified Numeric.LinearAlgebra as LA
import Numeric.LinearAlgebra((<>), (<.>), (|||))
import qualified Data.Vector.Storable as V

import qualified MachineLearning as ML
import MachineLearning.Types (R, Vector, Matrix)
import MachineLearning.Model


-- | Softmax Model, takes number of futures.
data SoftmaxModel = Softmax Int

-- | Multiclass Softmax Model with linear hypothesis function and Softmax cost (loss) function
instance Model SoftmaxModel where
  hypothesis (Softmax nLabels) x theta = LA.vector predictions
    where thetas = unflatten nLabels theta
          scores = calcScores x thetas
          scores' = LA.toRows scores
          predictions = map (fromIntegral . LA.maxIndex) scores'

  cost (Softmax nLabels) lambda x y theta =
    let nSamples = fromIntegral $ LA.rows x
        thetas = unflatten nLabels theta
        scores = calcScores x thetas
        sum_probs = reduceByRows V.sum $ exp scores
        loss = LA.sumElements $ (log sum_probs) - remap scores y
        thetaReg = V.slice 1 ((LA.cols x)-1) theta
        reg = ((thetaReg <.> thetaReg) * 0.5 * lambda)
    in (loss + reg) / nSamples 

  gradient (Softmax nLabels) lambda x y theta =
    let nSamples = fromIntegral $ LA.rows x
        thetas = unflatten nLabels theta
        ys = processOutput nLabels y
        scores = calcScores x thetas
        sum_probs = reduceByRows V.sum $ exp scores
        probs = (exp scores) / sum_probs
        probs' = probs - ys
        dw =  (LA.tr probs') <> x
        thetasReg = 0 ||| (ML.removeBiasDimension thetas)
        reg = ((LA.scalar lambda) * thetasReg)
    in LA.flatten $ (dw + reg)/ nSamples


remap :: Matrix -> Vector -> Matrix
remap m v = LA.remap cols rows m
  where cols = LA.asColumn $ V.fromList [0..(fromIntegral $ LA.rows m)-1]
        rows = LA.toInt $ LA.asColumn v

reduceByRows :: (Vector -> R) -> Matrix -> Matrix
reduceByRows f = LA.asColumn . LA.vector . map f . LA.toRows


sumByRows :: Matrix -> Matrix
sumByRows = reduceByRows V.sum


calcScores :: Matrix -> Matrix -> Matrix
calcScores x theta = scores - reduceByRows V.maximum scores
  where scores = x <> (LA.tr theta)


processOutput :: Int -> Vector -> Matrix
processOutput numLabels y = LA.fromColumns $ map f [0 .. numLabels-1]
  where f sample = V.map (\a -> if round a == sample then 1 else 0) y


unflatten :: Int -> Vector -> Matrix
unflatten nLabels v = LA.reshape cols v
  where cols = (V.length v) `div` nLabels
