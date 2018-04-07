{-# LANGUAGE RankNTypes #-}
{-|
Module: MachineLearning.NeuralNetwork.Layer
Description: Neural Network's Layer
Copyright: (c) Alexander Ignatyev, 2017-2018.
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


import Prelude hiding ((<>))
import MachineLearning.Types (R, Matrix)
import MachineLearning.Utils (sumByColumns)
import qualified Numeric.LinearAlgebra as LA
import Numeric.LinearAlgebra ((<>))
import qualified Control.Monad.Random as RndM


data Cache = Cache {
  cacheZ :: Matrix
  , cacheX :: Matrix
  , cacheW :: Matrix
  };


data Layer = Layer {
  lUnits :: Int
  , lForward :: Matrix -> Matrix -> Matrix -> Matrix
  , lBackward :: Matrix -> Cache -> (Matrix, Matrix, Matrix)
  , lActivation :: Matrix -> Matrix
  , lActivationGradient :: Matrix -> Matrix -> Matrix
  , lInitializeThetaM :: forall g. RndM.RandomGen g => (Int, Int) -> RndM.Rand g (Matrix, Matrix)
  }


affineForward :: Matrix -> Matrix -> Matrix -> Matrix
affineForward x b w = (x <> LA.tr w) + b


affineBackward delta (Cache _ x w) = (dx, db, dw)
  where m = fromIntegral $ LA.rows x
        dx = delta <> w
        db = (sumByColumns delta)/m
        dw = (LA.tr delta <> x)/m
