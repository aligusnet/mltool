{-|
Module: MachineLearning.NeuralNetwork.Topology
Description: Neural Network's Topology
Copyright: (c) Alexander Ignatyev, 2017
License: BSD-3
Stability: experimental
Portability: POSIX

Neural Network's Topology
-}

module MachineLearning.NeuralNetwork.Topology
(
  Topology
  , LossFunc(..)
  , makeTopology
  , loss
  , propagateForward
  , propagateBackward
  , numberOutputs
  , initializeTheta
  , initializeThetaIO
  , initializeThetaM
  , flatten
  , unflatten
)

where

import Control.Monad (zipWithM)
import Data.List (foldl')
import qualified Control.Monad.Random as RndM
import qualified Data.Vector.Storable as V
import qualified Numeric.LinearAlgebra as LA
import MachineLearning.Types (R, Vector, Matrix)
import MachineLearning.Utils (listOfTuplesToList)
import MachineLearning.NeuralNetwork.Layer (Layer(..), Cache(..))
import MachineLearning.NeuralNetwork.Regularization (Regularization, forwardReg, backwardReg)


-- | Loss function's type.
-- Takes x, weights and y.
type LossFunc = Matrix -> Matrix -> R


-- | Neural network topology has at least 2 elements: numver of input and number of outputs.
-- And sizes of hidden layers between 2 elements.
data Topology = Topology [(Int, Int)] [Layer] LossFunc


-- | Makes Neural Network's Topology.
-- Takes number of inputs, list of hidden layers, output layer and loss function.
makeTopology :: Int -> [Layer] -> Layer -> LossFunc -> Topology
makeTopology nInputs hiddenLayers outputLayer lossFunc =
  let layers = hiddenLayers ++ [outputLayer]
      layerSizes = nInputs : (map lUnits layers)
      sizes = getThetaSizes layerSizes
  in Topology sizes layers lossFunc
      

-- | Calculates loss for the given topology.
-- Takes topology, regularization, x, weights, y.
loss :: Topology -> Regularization -> Matrix -> [(Matrix, Matrix)] -> Matrix -> R
loss (Topology _ _ lf) reg x weights y =
  let lossValue = lf x y
      regValue = forwardReg reg weights
  in lossValue + regValue


-- | Implementation of forward propagation algorithm.
propagateForward :: Topology -> Matrix -> [(Matrix, Matrix)] -> (Matrix, [Cache])
propagateForward (Topology _ layers _) x thetaList = foldl' f (x, []) $ zip thetaList layers
  where f (a, cs) (theta, hl) =
          let (a', cache) = forwardPass hl a theta
          in (a', cache:cs)


-- | Makes one forward step for the given layer.
forwardPass :: Layer -> Matrix -> (Matrix, Matrix) -> (Matrix, Cache)
forwardPass layer a (b, w) = (a', Cache z a w)
  where z = lForward layer a b w
        a' = lActivation layer z


-- | Implementation of backward propagation algorithm.
propagateBackward :: Topology -> Regularization -> Matrix -> [Cache] -> Matrix -> [(Matrix, Matrix)]
propagateBackward (Topology _ layers _) reg scores (cache:cacheList) y = gradientList
  where cache' = Cache scores (cacheX cache) (cacheW cache)
        cacheList' = cache':cacheList
        gradientList = snd $ foldl' f (y, []) $ zip cacheList' $ reverse layers
        f (da, grads) (cache, hl) =
          let (da', db, dw) = backwardPass hl reg da cache
          in (da', (db, dw):grads)


-- | Makes one backward step for the given layer.
backwardPass :: Layer -> Regularization -> Matrix -> Cache -> (Matrix, Matrix, Matrix)
backwardPass layer reg da cache = (da', db, dw')
  where delta = lActivationGradient layer (cacheZ cache) da
        (da', db, dw) = lBackward layer delta cache
        dw' = dw + (backwardReg reg (cacheW cache))


-- | Returns number of outputs of the given topology.
numberOutputs :: Topology -> Int
numberOutputs (Topology nnt _ _) = fst $ last nnt


-- | Returns dimensions of weight matrices for given neural network topology
getThetaSizes :: [Int] -> [(Int, Int)]
getThetaSizes nn = zipWith (\r c -> (r, c)) (tail nn) nn


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
initializeThetaM :: RndM.RandomGen g => Topology -> RndM.Rand g Vector
initializeThetaM topology = flatten <$> initializeThetaListM topology


-- | Create and initialize list of weights matrices with random values
-- for given neural network topology.
initializeThetaListM :: RndM.RandomGen g => Topology -> RndM.Rand g [(Matrix, Matrix)]
initializeThetaListM (Topology sizes layers _) = zipWithM lInitializeThetaM layers sizes


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
