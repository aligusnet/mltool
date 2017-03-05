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
  , initializeTheta
  , initializeThetaIO
  , initializeThetaM

)

where


import qualified Data.Vector.Storable as V
import qualified Numeric.LinearAlgebra as LA
import System.Random (RandomGen)
import qualified Control.Monad.Random as RndM
import MachineLearning.Types (R, Vector, Matrix)
import qualified MachineLearning.LogisticModel as LM
import MachineLearning.Random
import MachineLearning.NeuralNetwork.Topology (Topology(..))
import MachineLearning.NeuralNetwork.Layer (Layer(..), affineForward, affineBackward)


-- | Creates toplogy. Takes number of inputs, number of outputs and list of hidden layers.
makeTopology :: Int -> Int -> [Int] -> Topology
makeTopology nInputs nOutputs hlUnits = Topology sizes layers loss
  where hiddenLayers = take (length hlUnits) $ repeat mkAffineSigmoidLayer
        outputLayer = mkSigmoidOutputLayer
        layers = hiddenLayers ++ [outputLayer]
        layerSizes = nInputs : (hlUnits ++ [nOutputs])
        sizes = getThetaSizes layerSizes


mkAffineSigmoidLayer = Layer {
  lForward = affineForward
  , lActivation = LM.sigmoid
  , lBackward = affineBackward
  , lActivationGradient = \z da -> da * LM.sigmoidGradient z
  }


mkSigmoidOutputLayer = Layer {
  lForward = affineForward
  , lActivation = LM.sigmoid
  , lBackward = affineBackward
  , lActivationGradient = \scores y -> scores - y
  }


-- | Create and initialize weights vector with random values
-- for given neural network topology.
-- Takes a seed to initialize generator of random numbers as a first parameter.
initializeTheta :: Int -> Topology -> Vector
initializeTheta seed topology = RndM.evalRand (initializeThetaM topology) gen
  where gen = RndM.mkStdGen seed


-- | Create and initialize weights vector with random values
-- for given neural network topology inside IO Monad.
initializeThetaIO :: Topology -> IO Vector
initializeThetaIO = RndM.evalRandIO . initializeThetaM


-- | Create and initialize weights vector with random values
-- for given neural network topology inside RandomMonad.
initializeThetaM :: RandomGen g => Topology -> RndM.Rand g Vector
initializeThetaM topology = V.concat <$> initializeThetaListM topology


-- | Create and initialize list of weights matrices with random values
-- for given neural network topology.
initializeThetaListM :: RandomGen g => Topology -> RndM.Rand g [Vector]
initializeThetaListM (Topology sizes _ _) = concat <$> mapM initTheta sizes
  where initTheta (r, c) = do
          let b :: Vector
              b = LA.konst 0 r
              eps = calcEps r c
          sequence [return b, getRandomRVectorM (r*c) (-eps, eps)]
        calcEps r c = (sqrt 6) / (sqrt . fromIntegral $ r + c)


-- | Returns dimensions of weight matrices for given neural network topology
getThetaSizes :: [Int] -> [(Int, Int)]
getThetaSizes nn = zipWith (\r c -> (r, c)) (tail nn) nn


-- Sigmoid Loss function
loss :: Matrix -> [(Matrix, Matrix)] -> Matrix -> R
loss x thetaList y = (LA.sumElements $ (-y) * log(tau + x) - (1-y) * log ((1+tau)-x))/m
  where tau = 1e-7
        m = fromIntegral $ LA.rows x
