{-|
Module: MachineLearning.NeuralNetwork.Regularization
Description: Regularization
Copyright: (c) Alexander Ignatyev, 2017
License: BSD-3
Stability: experimental
Portability: POSIX

Regularization.
-}

module MachineLearning.NeuralNetwork.Regularization
(
  Regularization(..)
  , forwardReg
  , backwardReg
)

where


import MachineLearning.Types (R, Matrix)
import qualified Numeric.LinearAlgebra as LA
import MachineLearning.Regularization (Regularization(..))


-- | Calcaulates regularization for forward propagation.
-- It takes regularization parameter and theta list.
forwardReg :: Regularization -> [(Matrix, Matrix)] -> R
forwardReg RegNone _ = 0
forwardReg (L2 lambda) thetaList = 0.5 * lambda * (sum $ map LA.norm_2 $ snd $ unzip thetaList)


-- | Calculates regularization for step of backward propagation.
-- It takes regularization parameter and theta.
backwardReg :: Regularization -> Matrix -> Matrix
backwardReg RegNone _ = 0
backwardReg (L2 lambda) w = w * (LA.scalar lambda)
