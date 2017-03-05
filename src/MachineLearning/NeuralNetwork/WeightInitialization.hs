{-|
Module: MachineLearning.NeuralNetwork.WeightInitialization
Description: Weight Initialization
Copyright: (c) Alexander Ignatyev, 2017
License: BSD-3
Stability: experimental
Portability: POSIX

Various Weight Initialization algorithms.
-}

module MachineLearning.NeuralNetwork.WeightInitialization
(
  nguyen
  , he
)

where


import qualified Numeric.LinearAlgebra as LA
import qualified Control.Monad.Random as RndM
import MachineLearning.Types (Matrix)
import MachineLearning.Random (getRandomRMatrixM)



-- | Weight Initialization Algorithm discussed in Nguyen at al.: https://web.stanford.edu/class/ee373b/nninitialization.pdf
-- Nguyen, Derrick, Widrow, B. Improving the learning speed of 2-layer neural networks by choosing initial values of adaptive weights.
-- In Proc. IJCNN, 1990; 3: 21-26.
nguyen :: RndM.RandomGen g => (Int, Int) -> RndM.Rand g (Matrix, Matrix)
nguyen (r, c) = do
  let b = LA.konst 0 (1, r)
      eps = (sqrt 6) / (sqrt . fromIntegral $ r + c)
  w <- getRandomRMatrixM r c (-eps, eps)
  return (b, w)


-- | Weight Initialization Algorithm discussed in He at al.: https://arxiv.org/abs/1502.01852
-- Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
-- Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification.
he :: RndM.RandomGen g => (Int, Int) -> RndM.Rand g (Matrix, Matrix)
he (r, c) = do
  let b = LA.konst 0 (1, r)
      eps = sqrt (2/(fromIntegral $ r + c))
  w <- getRandomRMatrixM r c (-eps, eps)
  return (b, w)
