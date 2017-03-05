{-|
Module: MachineLearning.NeuralNetwork
Description: Neural Network
Copyright: (c) Alexander Ignatyev, 2016
License: BSD-3
Stability: experimental
Portability: POSIX

Simple Neural Networks.
-}

module MachineLearning.NeuralNetwork
(
    Model(..)
  , NeuralNetworkModel(..)
  , Topology
  , MLC.calcAccuracy

  -- * Exported for testing purposes only.
  , flatten
  , unflatten
)

where

import Data.List (foldl')
import qualified Data.Vector.Storable as V
import qualified Numeric.LinearAlgebra as LA
import MachineLearning.Types (R, Vector, Matrix)
import qualified MachineLearning as ML
import qualified MachineLearning.Classification.Internal as MLC
import MachineLearning.Model (Model(..))
import MachineLearning.NeuralNetwork.Topology (Topology(..), loss, propagateForward, propagateBackward, numberOutputs)
import MachineLearning.NeuralNetwork.Regularization (Regularization(L2))


-- | Neural Network Model.
-- Takes neural network topology as a constructor argument.
newtype NeuralNetworkModel = NeuralNetwork Topology


instance Model NeuralNetworkModel where
  hypothesis (NeuralNetwork topology) x theta = predictions'
    where thetaList = unflatten topology theta
          predictions = LA.toRows $ calcScores topology x thetaList
          predictions' = LA.vector $ map (fromIntegral . LA.maxIndex) predictions

  cost (NeuralNetwork topology) lambda x y theta = 
    let ys = LA.fromColumns $ MLC.processOutputOneVsAll (numberOutputs topology) y
        thetaList = unflatten topology theta
        scores = calcScores topology x thetaList
    in loss topology (L2 lambda) scores thetaList ys

  gradient (NeuralNetwork topology) lambda x y theta =
    let ys = LA.fromColumns $ MLC.processOutputOneVsAll (numberOutputs topology) y
        thetaList = unflatten topology theta
        (scores, cacheList) = propagateForward topology x thetaList
        grad = flatten $ propagateBackward topology (L2 lambda) scores cacheList ys
    in grad


-- | Score function. Takes a topology, X and theta list.
calcScores :: Topology -> Matrix -> [(Matrix, Matrix)] -> Matrix
calcScores topology x thetaList = fst $ propagateForward topology x thetaList


-- | Flatten list of matrices into vector.
flatten :: [(Matrix, Matrix)] -> Vector
flatten ms = V.concat $ map LA.flatten $ listOfTuplesToList ms


-- | Unflatten vector into list of matrices for given neural network topology.
unflatten :: Topology -> Vector -> [(Matrix, Matrix)]
unflatten (Topology sizes _ _) v =
  let offsets = reverse $ foldl' (\os (r, c) -> (r+r*c + head os):os) [0] (init sizes)
      ms = zipWith (\o (r, c) -> (LA.reshape r (slice o r), LA.reshape c (slice (o+r) (r*c)))) offsets sizes
      slice o n = V.slice o n v
  in ms


listOfTuplesToList [] = []
listOfTuplesToList ((a, b):xs) = a : b : listOfTuplesToList xs
