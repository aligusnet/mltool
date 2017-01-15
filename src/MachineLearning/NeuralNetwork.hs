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
  NeuralNetworkModel(..)
  , Topology
  , makeTopology
  , initializeTheta
  , initializeThetaIO
  , initializeThetaM
  , predictMulti
  , MLC.calcAccuracy

  -- * exported for testing purposes only
  , flatten
  , unflatten
  , getThetaSizes
  , getThetaTotalSize
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
import Types (R, Vector, Matrix)
import qualified MachineLearning as ML
import qualified MachineLearning.Regression.Logistic as LR
import qualified MachineLearning.Classification as MLC
import MachineLearning.Regression.Model (Model(..))
import MachineLearning.Random


-- | Neural network topology has at least 2 elements: numver of input and number of outputs.
-- And sizes of hidden layers between 2 elements.
-- Bias input must not be included.
newtype Topology = Topology [Int]


-- | Creates toplogy. Takes number of inputs, number of outputs and list of hidden layers.
makeTopology :: Int -> Int -> [Int] -> Topology
makeTopology nInputs nOutputs hiddenLayers = Topology $ nInputs : (hiddenLayers ++ [nOutputs])


-- | Returns number of outputs of the given topology.
numberOutputs :: Topology -> Int
numberOutputs (Topology nnt) = last nnt


-- | Neural Network Model.
-- Takes neural network topology as a constructor argument.
newtype NeuralNetworkModel = NeuralNetwork Topology

instance Model NeuralNetworkModel where
  hypothesis (NeuralNetwork topology) x theta =
    let thetaList = unflatten topology theta
    in LA.flatten $ calcLastActivation x thetaList

  cost (NeuralNetwork topology) lambda x y theta = 
    let ys = LA.fromColumns $ MLC.processOutputMulti (numberOutputs topology) y
        thetaList = unflatten topology theta
    in calculateCost lambda x thetaList ys

  gradient (NeuralNetwork topology) lambda x y theta =
    let ys = LA.fromColumns $ MLC.processOutputMulti (numberOutputs topology) y
        thetaList = unflatten topology theta
        (activationList, zList) = propagateForward x thetaList
        grad = flatten $ propagateBackward lambda activationList zList thetaList ys
    in grad


-- | Flatten list of matrices into vector.
flatten :: [Matrix] -> Vector
flatten ms = V.concat $ map LA.flatten ms


-- | Unflatten vector into list of matrices for given neural network topology.
unflatten :: Topology -> Vector -> [Matrix]
unflatten topology v =
  let sizes = getThetaSizes topology
      offsets = reverse $ foldl' (\os (r, c) -> (r*c + head os):os) [0] (init sizes)
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
initializeThetaListM (Topology nn) = zipWithM initTheta (tail nn) nn
  where initTheta r c = do
          let eps = calcEps r c
          getRandomRMatrixM r (c+1) (-eps, eps)
        calcEps r c = (sqrt 6) / (sqrt . fromIntegral $ r + c)


-- | Return sum of dimensions of weight matrices for given neural network topology.
getThetaTotalSize :: Topology -> Int
getThetaTotalSize topology = sum $ map (\(c, r) -> c*r) $ getThetaSizes topology


-- | Returns dimensions of weight matrices for given neural network topology
getThetaSizes :: Topology -> [(Int, Int)]
getThetaSizes (Topology nn) = zipWith (\r c -> (r, c+1)) (tail nn) nn


-- | Calculates last layer of activation units
calcLastActivation :: Matrix -> [Matrix] -> Matrix
calcLastActivation x thetaList = head . fst $ propagateForward x thetaList


-- | Calculate output for given input matrix (X) and list of weight vector (theta).
-- Returns vector of predicted class indices (assuming class indices starts at 0).
predictMulti :: Topology -> Matrix -> Vector -> Vector
predictMulti topology x theta = predictions'
  where thetaList = unflatten topology theta
        predictions = LA.toRows $ (calcLastActivation x thetaList) LA.?? (LA.All, LA.Drop 1)
        predictions' = LA.vector $ map (fromIntegral . LA.maxIndex) predictions


-- | Calculats activations and z, returns them in reverse order.
propagateForward :: Matrix -> [Matrix] -> ([Matrix], [Matrix])
propagateForward x thetaList = foldl' f ([x], []) thetaList
  where f :: ([Matrix], [Matrix]) -> Matrix -> ([Matrix], [Matrix])
        f (al, zl) theta =
          let z = (head al) <> LA.tr theta
              a = ML.addColumnOfOnes $ LR.sigmoid z
          in (a:al, z:zl)


-- | Used as helping procedure for Model.hypothesis
calculateCost :: R -> Matrix -> [Matrix] -> Matrix -> R
calculateCost lambda x thetaList y = (LA.sumElements $ (-y) * log(h) - (1-y) * log (1-h))/m + reg'
  where h = (calcLastActivation x thetaList) LA.?? (LA.All, LA.Drop 1)
        m = fromIntegral $ LA.rows x
        reg = sum $ map (\t -> LA.norm_2 $ t LA.?? (LA.All, LA.Drop 1) ) thetaList
        reg' = reg * lambda * 0.5 / m


-- | Implements backward propagation algorithm.
propagateBackward :: R -> [Matrix] -> [Matrix] -> [Matrix] -> Matrix -> [Matrix]
propagateBackward lambda activationList zList thetaList y = reverse gradientList
  where m = fromIntegral $ LA.rows y
        thetaList' = reverse $ map (\t -> t LA.?? (LA.All, LA.Drop 1)) thetaList
        deltaLast :: Matrix
        deltaLast = ((head activationList) LA.?? (LA.All, LA.Drop 1)) - y
        deltaList :: [Matrix]
        deltaList = foldl' f [deltaLast] $ zip (tail zList) thetaList'
        f :: [Matrix] -> (Matrix, Matrix) -> [Matrix]
        f dList (z, theta) = ((head dList <> theta) * (LR.sigmoidGradient z)) : dList
        gradientList = zipWith3 (\d a t-> ((LA.tr d <> a) + ((0 ||| t) * (LA.scalar lambda))) / m) (reverse deltaList) (tail activationList) thetaList'
