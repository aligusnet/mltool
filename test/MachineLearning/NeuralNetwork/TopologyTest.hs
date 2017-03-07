module MachineLearning.NeuralNetwork.TopologyTest
(
  tests
)

where

import Test.Framework (testGroup)
import Test.Framework.Providers.HUnit
import Test.HUnit
import Test.HUnit.Approx
import Test.HUnit.Plus
import qualified Numeric.LinearAlgebra as LA
import MachineLearning.NeuralNetwork.Topology
import qualified MachineLearning.NeuralNetwork.TopologyMaker as TM

nnt = TM.makeTopology TM.ASigmoid TM.LLogistic 15 2 [10]

flattenTest = do
  theta <- initializeThetaIO nnt
  let theta' = flatten $ unflatten nnt theta
      norm = LA.norm_2 (theta - theta')
  assertApproxEqual "flatten" 1e-10 0 norm

tests = [ testGroup "flatten" [
            testCase "flatten" flattenTest
            ]
        ]
