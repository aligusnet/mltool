module MachineLearning.NeuralNetwork.WeightInitializationTest
(
  tests
)

where

import Test.Framework (testGroup)
import Test.Framework.Providers.HUnit
import Test.HUnit
import Test.HUnit.Approx
import Test.HUnit.Plus
import Control.Monad (replicateM)
import qualified Data.Vector.Storable as V
import qualified Numeric.LinearAlgebra as LA
import qualified Numeric.GSL.Statistics as Stat
import qualified Control.Monad.Random as RndM
import MachineLearning.NeuralNetwork.WeightInitialization

assertDistribution eps x mu e = do
  let mean = Stat.mean x
  assertApproxEqual "mean" eps mu mean
  assertBool "maximum" $ (V.maximum x) <= e
  assertBool "minimum" $ (V.minimum x) >= (-e)


generateData algo n sz = do
   rndList <- replicateM n $ RndM.evalRandIO (snd <$> algo sz)
   return . LA.flatten $ LA.fromBlocks [rndList]


testWeightInitAlgo eps algo n sz mu e = do
  m <- generateData algo n sz
  assertDistribution eps m mu e


heEps (r, c) = sqrt (2/(fromIntegral $ r + c))
nguyenEps (r, c) = (sqrt 6) / (sqrt . fromIntegral $ r + c)

sz = (7, 5)

tests = [ testGroup "flatten" [
            testCase "hu" $ testWeightInitAlgo 1e-1 he 100 sz 0 (heEps sz)
            , testCase "nguyen" $ testWeightInitAlgo 1e-1 nguyen 100 sz 0 (nguyenEps sz)
            ]
        ]
