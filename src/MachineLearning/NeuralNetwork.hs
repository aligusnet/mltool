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
  , MLC.calcAccuracy
  , T.Topology
  , T.initializeTheta
  , T.initializeThetaIO
  , T.initializeThetaM
)

where

import Data.List (foldl')
import qualified Numeric.LinearAlgebra as LA
import MachineLearning.Types (R, Vector, Matrix)
import qualified MachineLearning as ML
import qualified MachineLearning.Classification.Internal as MLC
import MachineLearning.Model (Model(..))
import qualified MachineLearning.NeuralNetwork.Topology as T
import MachineLearning.NeuralNetwork.Regularization (Regularization(L2))


-- | Neural Network Model.
-- Takes neural network topology as a constructor argument.
newtype NeuralNetworkModel = NeuralNetwork T.Topology


instance Model NeuralNetworkModel where
  hypothesis (NeuralNetwork topology) x theta = predictions'
    where thetaList = T.unflatten topology theta
          predictions = LA.toRows $ calcScores topology x thetaList
          predictions' = LA.vector $ map (fromIntegral . LA.maxIndex) predictions

  cost (NeuralNetwork topology) lambda x y theta = 
    let ys = LA.fromColumns $ MLC.processOutputOneVsAll (T.numberOutputs topology) y
        thetaList = T.unflatten topology theta
        scores = calcScores topology x thetaList
    in T.loss topology (L2 lambda) scores thetaList ys

  gradient (NeuralNetwork topology) lambda x y theta =
    let ys = LA.fromColumns $ MLC.processOutputOneVsAll (T.numberOutputs topology) y
        thetaList = T.unflatten topology theta
        (scores, cacheList) = T.propagateForward topology x thetaList
        grad = T.flatten $ T.propagateBackward topology (L2 lambda) scores cacheList ys
    in grad


-- | Score function. Takes a topology, X and theta list.
calcScores :: T.Topology -> Matrix -> [(Matrix, Matrix)] -> Matrix
calcScores topology x thetaList = fst $ T.propagateForward topology x thetaList
