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
import qualified Numeric.Morpheus.Statistics as Stat
import qualified Control.Monad.Random as RndM
import MachineLearning.NeuralNetwork.WeightInitialization

assertDistribution eps x mu e = do
  let mean = Stat.mean x
  assertApproxEqual "mean" eps mu mean
  assertBool "maximum" $ (V.maximum x) <= e
  assertBool "minimum" $ (V.minimum x) >= (-e)


generateData algo n sz = do
   rndList <- replicateM n $ RndM.evalRandIO (algo sz)
   let (bs, ws) = unzip rndList
       b = LA.flatten $ LA.fromBlocks [bs]
       w = LA.flatten $ LA.fromBlocks [ws]
   return (b, w)


testWeightInitAlgo eps algo n sz mu e = do
  (b, w) <- generateData algo n sz
  assertDistribution eps b 0 0
  assertDistribution eps w mu e


heEps (r, c) = sqrt (2/(fromIntegral $ r + c))
nguyenEps (r, c) = (sqrt 6) / (sqrt . fromIntegral $ r + c)

sz = (7, 5)

tests = [ testGroup "flatten" [
            testCase "hu" $ testWeightInitAlgo 1e-1 he 100 sz 0 (heEps sz)
            , testCase "nguyen" $ testWeightInitAlgo 1e-1 nguyen 100 sz 0 (nguyenEps sz)
            ]
        ]
