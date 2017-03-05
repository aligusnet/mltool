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
  Topology(..)
  , loss
  , propagateForward
  , propagateBackward
  , numberOutputs
)

where

import Data.List (foldl')
import MachineLearning.Types (R, Matrix)
import MachineLearning.NeuralNetwork.Layer (Layer(..), Cache(..))
import MachineLearning.NeuralNetwork.Regularization (Regularization, forwardReg, backwardReg)

type LossFunc = Matrix -> [(Matrix, Matrix)] -> Matrix -> R


-- | Neural network topology has at least 2 elements: numver of input and number of outputs.
-- And sizes of hidden layers between 2 elements.
data Topology = Topology [(Int, Int)] [Layer] LossFunc


loss :: Topology -> Regularization -> Matrix -> [(Matrix, Matrix)] -> Matrix -> R
loss (Topology _ _ lf) reg x weights y =
  let lossValue = lf x weights y
      regValue = forwardReg reg weights
  in lossValue + regValue


propagateForward (Topology _ layers _) x thetaList = foldl' f (x, []) $ zip thetaList layers
  where f (a, cs) (theta, hl) =
          let (a', cache) = forwardPass hl a theta
          in (a', cache:cs)


forwardPass layer a (b, w) = (a', Cache z a b w)
  where z = lForward layer a b w
        a' = lActivation layer z


-- | Implements backward propagation algorithm.
propagateBackward (Topology _ layers _) reg scores (cache:cacheList) y = gradientList
  where cache' = Cache scores (cacheX cache) (cacheB cache) (cacheW cache)
        cacheList' = cache':cacheList
        gradientList = snd $ foldl' f (y, []) $ zip cacheList' $ reverse layers
        f (da, grads) (cache, hl) =
          let (da', db, dw) = backwardPass hl reg da cache
          in (da', (db, dw):grads)


backwardPass layer reg da cache = (da', db, dw')
  where delta = lActivationGradient layer (cacheZ cache) da
        (da', db, dw) = lBackward layer delta cache
        dw' = dw + (backwardReg reg (cacheW cache))


-- | Returns number of outputs of the given topology.
numberOutputs :: Topology -> Int
numberOutputs (Topology nnt _ _) = fst $ last nnt
