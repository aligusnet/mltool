module MachineLearningTest
(
  tests
)

where


import Test.Framework (testGroup)
import Test.Framework.Providers.HUnit
import Test.HUnit
import Test.HUnit.Plus

import MachineLearning

import qualified Numeric.LinearAlgebra as LA

x = LA.matrix 3 [17, 19, 29]
x2 = LA.matrix 9 [17, 19, 29, 289, 323, 493, 361, 551, 841]
x3 = LA.matrix 19 [17, 19, 29, 289, 323, 493, 361, 551, 841, 4913, 5491, 8381, 6137, 9367, 14297, 6859, 10469, 15979, 24389]
eps = 1e-5

tests = [testGroup "mapFeatures" [
            testCase "mapfeatures 1" $ assertMatrix "" eps x (mapFeatures 1 x)
            , testCase "mapfeatures 2" $ assertMatrix "" eps x2 (mapFeatures 2 x)
            , testCase "mapfeatures 3" $ assertMatrix "" eps x3 (mapFeatures 3 x)
            ]
        ]
