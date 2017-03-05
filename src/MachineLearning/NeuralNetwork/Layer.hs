{-|
Module: MachineLearning.NeuralNetwork.Layer
Description: Neural Network's Layer
Copyright: (c) Alexander Ignatyev, 2017
License: BSD-3
Stability: experimental
Portability: POSIX

Neural Network's Layer
-}


module MachineLearning.NeuralNetwork.Layer
(
  Layer(..)
  , Cache(..)
  , affineForward
  , affineBackward
)

where


import MachineLearning.Types (R, Matrix)
import qualified Data.Vector.Storable as V
import qualified Numeric.LinearAlgebra as LA
import Numeric.LinearAlgebra ((<>))


data Cache = Cache {
  cacheZ :: Matrix
  , cacheX :: Matrix
  , cacheB :: Matrix
  , cacheW :: Matrix
  };


data Layer = Layer {
  lForward :: Matrix -> Matrix -> Matrix -> Matrix
  , lBackward :: Matrix -> Cache -> (Matrix, Matrix, Matrix)
  , lActivation :: Matrix -> Matrix
  , lActivationGradient :: Matrix -> Matrix -> Matrix
  }


affineForward :: Matrix -> Matrix -> Matrix -> Matrix
affineForward x b w = (x <> LA.tr w) + b


affineBackward delta (Cache _ x b w) = (dx, db, dw)
  where m = fromIntegral $ LA.rows x
        dx = delta <> w
        db = (sumByCols delta)/m
        dw = (LA.tr delta <> x)/m


sumByCols :: Matrix -> Matrix
sumByCols x = LA.asRow . LA.vector $ map V.sum $ LA.toColumns x
