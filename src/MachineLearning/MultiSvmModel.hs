{-|
Module: MachineLearning.MultiSvmModel
Description: Multiclass Support Vector Machines Model.
Copyright: (c) Alexander Ignatyev, 2017.
License: BSD-3
Stability: experimental
Portability: POSIX

Multicalss Support Vector Machines Model.
-}

module MachineLearning.MultiSvmModel
(
  module MachineLearning.Model
  , MultiSvmModel(..)
)

where

import qualified Numeric.LinearAlgebra as LA
import Numeric.LinearAlgebra((<>), (<.>), (|||))
import qualified Data.Vector.Storable as V

import qualified MachineLearning as ML
import MachineLearning.Types (R, Vector, Matrix)
import MachineLearning.Model


-- | Multiclass SVM Model, takes delta and number of futures. Delta = 1.0 is good for all cases.
data MultiSvmModel = MultiSvm R Int

-- | Multiclass SVM Model with linear hypothesis function and SVM cost (loss) function
instance Model MultiSvmModel where
  hypothesis (MultiSvm _ nLabels) x theta = LA.vector predictions
    where thetas = unflatten nLabels theta
          scores = calcScores x thetas
          scores' = LA.toRows scores
          predictions = map (fromIntegral . LA.maxIndex) scores'

  cost (MultiSvm d nLabels) lambda x y theta =
    let nSamples = fromIntegral $ LA.rows x
        thetas = unflatten nLabels theta
        scores = calcScores x thetas
        correct_scores = LA.remap (LA.asColumn $ V.fromList [0..(fromIntegral $ LA.rows x)-1]) (LA.toInt $ LA.asColumn y) scores
        margins = scores - (correct_scores - (LA.scalar d))
        margins' = LA.cmap (\e -> max 0 e) margins
        loss = LA.sumElements(margins') / nSamples - d
        thetaReg = V.slice 1 ((LA.cols x)-1) theta
        reg = ((thetaReg <.> thetaReg) * 0.5 * lambda) / nSamples
    in loss + reg

  gradient (MultiSvm d nLabels) lambda x y theta =
    let nSamples = fromIntegral $ LA.rows x
        thetas = unflatten nLabels theta
        ys = processOutput nLabels y
        scores = calcScores x thetas
        correct_scores = LA.remap (LA.asColumn $ V.fromList [0..(fromIntegral $ LA.rows x)-1]) (LA.toInt $ LA.asColumn y) scores
        margins = scores - (correct_scores - (LA.scalar d))
        margins' = (1-ys)*(LA.step margins)  -- step == cmap (\x -> if x>0 then 1 else 0)
        k = sumByRows margins'
        margins'' = margins' - (ys * k)
        dw = ((LA.tr margins'') <> x) / nSamples
        thetasReg = 0 ||| (ML.removeBiasDimension thetas)
        reg = ((LA.scalar lambda) * thetasReg) / nSamples
    in LA.flatten $ dw + reg


sumByRows :: Matrix -> Matrix
sumByRows x = LA.asColumn . LA.vector $ map V.sum $ LA.toRows x


calcScores :: Matrix -> Matrix -> Matrix
calcScores x theta = x <> (LA.tr theta)


processOutput :: Int -> Vector -> Matrix
processOutput numLabels y = LA.fromColumns $ map f [0 .. numLabels-1]
  where f sample = V.map (\a -> if round a == sample then 1 else 0) y


unflatten :: Int -> Vector -> Matrix
unflatten nLabels v = LA.reshape cols v
  where cols = (V.length v) `div` nLabels
