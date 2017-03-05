{-|
Module: MachineLearning.NeuralNetwork.Sigmoid
Description: Sigmoid
Copyright: (c) Alexander Ignatyev, 2017
License: BSD-3
Stability: experimental
Portability: POSIX

Sigmoid
-}

module MachineLearning.NeuralNetwork.Sigmoid
(
    makeTopology
)

where


import qualified Data.Vector.Storable as V
import qualified Numeric.LinearAlgebra as LA
import qualified Control.Monad.Random as RndM
import MachineLearning.Types (R, Vector, Matrix)
import qualified MachineLearning.LogisticModel as LM
import MachineLearning.Random (getRandomRMatrixM)
import qualified MachineLearning.NeuralNetwork.Topology as T
import MachineLearning.NeuralNetwork.Layer (Layer(..), affineForward, affineBackward)


-- | Creates toplogy. Takes number of inputs, number of outputs and list of hidden layers.
makeTopology :: Int -> Int -> [Int] -> T.Topology
makeTopology nInputs nOutputs hlUnits = T.makeTopology nInputs hiddenLayers outputLayer loss
  where hiddenLayers = map mkAffineSigmoidLayer hlUnits
        outputLayer = mkSigmoidOutputLayer nOutputs


mkAffineSigmoidLayer nUnits = Layer {
  lUnits = nUnits
  , lForward = affineForward
  , lActivation = LM.sigmoid
  , lBackward = affineBackward
  , lActivationGradient = \z da -> da * LM.sigmoidGradient z
  , lInitializeThetaM = initializeThetaM
  }


mkSigmoidOutputLayer nUnits = Layer {
  lUnits = nUnits
  , lForward = affineForward
  , lActivation = LM.sigmoid
  , lBackward = affineBackward
  , lActivationGradient = \scores y -> scores - y
  , lInitializeThetaM = initializeThetaM
  }


initializeThetaM :: RndM.RandomGen g => (Int, Int) -> RndM.Rand g (Matrix, Matrix)
initializeThetaM (r, c) = do
  let b = LA.konst 0 (1, r)
      eps = (sqrt 6) / (sqrt . fromIntegral $ r + c)
  w <- getRandomRMatrixM r c (-eps, eps)
  return (b, w)


-- Sigmoid Loss function
loss :: Matrix -> [(Matrix, Matrix)] -> Matrix -> R
loss x thetaList y = (LA.sumElements $ (-y) * log(tau + x) - (1-y) * log ((1+tau)-x))/m
  where tau = 1e-7
        m = fromIntegral $ LA.rows x
