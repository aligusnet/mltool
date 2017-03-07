module MachineLearning.Optimization.MinibatchGradientDescentTest
(
  tests
)

where

import Test.Framework (testGroup)
import Test.Framework.Providers.HUnit
import Test.HUnit
import Test.HUnit.Approx
import Test.HUnit.Plus

import qualified Data.Vector.Storable as V
import qualified Numeric.LinearAlgebra as LA

import MachineLearning.Types (Vector)
import qualified MachineLearning as ML
import MachineLearning.Regularization (Regularization(..))
import MachineLearning.LeastSquaresModel (LeastSquaresModel(..))
import MachineLearning.Optimization.MinibatchGradientDescent

import MachineLearning.DataSets (dataset1)

(x, y) = ML.splitToXY dataset1

muSigma = ML.meanStddev x
xNorm = ML.featureNormalization muSigma x
x1 = ML.addBiasDimension xNorm
initialTheta :: Vector
initialTheta = LA.konst 0 (LA.cols x1)
lsExpectedTheta = LA.vector [325009.354,113890.981,6876.935]
eps = 1e-3


isInDescendingOrder :: [Double] -> Bool
isInDescendingOrder lst = and . snd . unzip $ scanl (\(prev, _) current -> (current, prev >= current)) (1/0, True) lst

testMinibatchGradientDescent model expectedTheta = do
  let (theta, optPath) = minibatchGradientDescent 0 16 0.01 model eps 5000 RegNone x1 y initialTheta
      js = V.toList $ (LA.toColumns optPath) !! 1
  assertVector "theta" 0.01 expectedTheta theta
  assertBool "non-increasing errors" $ isInDescendingOrder js

tests = [testGroup "minibatchGradientDescent" [
            testCase "leastSquares" $ testMinibatchGradientDescent LeastSquares lsExpectedTheta
            ]
        ]
