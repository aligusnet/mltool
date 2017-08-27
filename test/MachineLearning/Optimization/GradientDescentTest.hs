module MachineLearning.Optimization.GradientDescentTest
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

import qualified MachineLearning as ML
import MachineLearning.Regularization (Regularization(..))
import MachineLearning.LeastSquaresModel (LeastSquaresModel(..))
import MachineLearning.Optimization.GradientDescent

import MachineLearning.DataSets (dataset1)

(x, y) = ML.splitToXY dataset1

muSigma = ML.meanStddev x
xNorm = ML.featureNormalization muSigma x
x1 = ML.addBiasDimension xNorm
initialTheta = LA.konst 0 (LA.cols x1)
-- Normal Equation's Result: 340412.660,110631.050,-6649.474
lsExpectedTheta = LA.vector [340412.660, 110630.886, -6649.310]
eps = 1e-3


isInDescendingOrder :: [Double] -> Bool
isInDescendingOrder lst = and . snd . unzip $ scanl (\(prev, _) current -> (current, prev >= current)) (1/0, True) lst

testGradientDescent model expectedTheta = do
  let (theta, optPath) = gradientDescent 0.01 model eps 5000 RegNone x1 y initialTheta
      js = V.toList $ (LA.toColumns optPath) !! 1
  assertVector "theta" 0.01 expectedTheta theta
  assertBool "non-increasing errors" $ isInDescendingOrder js

tests = [testGroup "gradientDescent" [
            testCase "leastSquares" $ testGradientDescent LeastSquares lsExpectedTheta
            ]
        ]
