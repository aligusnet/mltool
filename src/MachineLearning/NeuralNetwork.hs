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
  , makeTopology
  , initializeTheta
  , initializeThetaIO
  , initializeThetaM
  , MLC.calcAccuracy

  -- * Exported for testing purposes only.
  , flatten
  , unflatten
  , initializeThetaListM
)

where

import Data.List (foldl')
import Control.Monad (zipWithM)
import qualified Data.Vector.Storable as V
import qualified Numeric.LinearAlgebra as LA
import Numeric.LinearAlgebra ((<>), (|||))
import System.Random (RandomGen)
import qualified Control.Monad.Random as RndM
import MachineLearning.Types (R, Vector, Matrix)
import qualified MachineLearning as ML
import qualified MachineLearning.LogisticModel as LM
import qualified MachineLearning.Classification.Internal as MLC
import MachineLearning.Model (Model(..))
import MachineLearning.Random


-- | Neural network topology has at least 2 elements: numver of input and number of outputs.
-- And sizes of hidden layers between 2 elements.
-- Bias input must not be included.
data Topology = Topology [(Int, Int)] [Layer]


-- | Creates toplogy. Takes number of inputs, number of outputs and list of hidden layers.
makeTopology :: Int -> Int -> [Int] -> Topology
makeTopology nInputs nOutputs hlUnits = Topology sizes layers
  where hiddenLayers = take (length hlUnits) $ repeat mkAffineSigmoidLayer
        outputLayer = mkSigmoidOutputLayer
        layers = hiddenLayers ++ [outputLayer]
        layerSizes = nInputs : (hlUnits ++ [nOutputs])
        sizes = getThetaSizes layerSizes


-- | Returns number of outputs of the given topology.
numberOutputs :: Topology -> Int
numberOutputs (Topology nnt _) = fst $ last nnt


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
    in sigmoidLoss (L2 lambda) scores thetaList ys

  gradient (NeuralNetwork topology) lambda x y theta =
    let ys = LA.fromColumns $ MLC.processOutputOneVsAll (numberOutputs topology) y
        thetaList = unflatten topology theta
        (scores, cacheList) = propagateForward topology x thetaList
        grad = flatten $ propagateBackward topology (L2 lambda) scores cacheList ys
    in grad



calcScores topology x thetaList = ML.removeBiasDimension $ fst $ propagateForward topology x thetaList

-- | Flatten list of matrices into vector.
flatten :: [Matrix] -> Vector
flatten ms = V.concat $ map LA.flatten ms


-- | Unflatten vector into list of matrices for given neural network topology.
unflatten :: Topology -> Vector -> [Matrix]
unflatten (Topology sizes _) v =
  let offsets = reverse $ foldl' (\os (r, c) -> (r*c + head os):os) [0] (init sizes)
      ms = zipWith (\o (r, c) -> LA.reshape c $ V.slice o (r*c) v) offsets sizes
  in ms


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
initializeThetaM topology = flatten <$> initializeThetaListM topology


-- | Create and initialize list of weights matrices with random values
-- for given neural network topology.
initializeThetaListM :: RandomGen g => Topology -> RndM.Rand g [Matrix]
initializeThetaListM (Topology sizes _) = mapM initTheta sizes
  where initTheta (r, c) = do
          let eps = calcEps r c
          getRandomRMatrixM r c (-eps, eps)
        calcEps r c = (sqrt 6) / (sqrt . fromIntegral $ r + c - 1)


-- | Returns dimensions of weight matrices for given neural network topology
getThetaSizes :: [Int] -> [(Int, Int)]
getThetaSizes nn = zipWith (\r c -> (r, c+1)) (tail nn) nn


data Regularization = L2 Double
forwardReg (L2 lambda) thetaList = 0.5 * lambda * (sum $ map (LA.norm_2 . ML.removeBiasDimension) thetaList)
backwardReg (L2 lambda) theta = (0 ||| ML.removeBiasDimension theta) * (LA.scalar lambda)


data Cache = Cache {
  cacheZ :: Matrix
  , cacheX :: Matrix
  , cacheTheta :: Matrix
  };


data Layer = Layer {
  lForward :: Matrix -> Matrix -> Matrix
  , lBackward :: Matrix -> Cache -> (Matrix, Matrix)
  , lActivation :: Matrix -> Matrix
  , lActivationGradient :: Matrix -> Matrix -> Matrix
  }


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
  , lActivationGradient = \scores y -> ML.removeBiasDimension scores - y
  }


propagateForward (Topology _ layers) x thetaList = foldl' f (x, []) $ zip thetaList layers
  where f (a, cs) (theta, hl) =
          let (a', cache) = forwardPass hl a theta
          in (a', cache:cs)


forwardPass layer a theta = (a', Cache z a theta)
  where z = lForward layer a theta
        a' = ML.addBiasDimension $ lActivation layer z


-- | Implements backward propagation algorithm.
propagateBackward (Topology _ layers) reg scores (cache:cacheList) y = gradientList
  where cache' = Cache scores (cacheX cache) (cacheTheta cache)
        cacheList' = cache':cacheList
        gradientList = snd $ foldl' f (y, []) $ zip cacheList' $ reverse layers
        f (da, grads) (cache, hl) =
          let (da', grad') = backwardPass hl reg da cache
          in (da', grad':grads)


backwardPass layer reg da cache = (da', grad')
  where delta = lActivationGradient layer (cacheZ cache) da
        (da', grad) = lBackward layer delta cache
        grad' = grad + (backwardReg reg (cacheTheta cache))

affineForward x theta = x <> LA.tr theta


affineBackward delta (Cache _ x theta) = (da, grad)
  where theta' = ML.removeBiasDimension theta
        m = fromIntegral $ LA.rows x
        grad = (LA.tr delta <> x)/m
        da = delta <> theta'


sigmoidLoss reg x thetaList y = (LA.sumElements $ (-y) * log(tau + x) - (1-y) * log ((1+tau)-x))/m + r
  where tau = 1e-7
        m = fromIntegral $ LA.rows x
        r = forwardReg reg thetaList
