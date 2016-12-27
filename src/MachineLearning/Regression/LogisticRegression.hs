{-|
Module: MachineLearning.Regression.Logistic
Description: Logistic Regression
Copyright: (c) Alexander Ignatyev, 2016
License: BSD-3
Stability: experimental
Portability: POSIX

-}

module MachineLearning.Regression.LogisticRegression
(
  LogisticModel(..)
)

where

import qualified Numeric.LinearAlgebra as LA
import Numeric.LinearAlgebra((<>), (#>))
import qualified Numeric.LinearAlgebra.Data as LAD

import MachineLearning.Regression.Model

data LeastSquaresModel = LeastSquares

instance Model LeastSquaresModel where
  hypothesis _ x theta = x #> theta

  cost _ x y theta = 
    let m = x #> theta - y
        nrows = fromIntegral $ LAD.rows x
    in (LA.sumElements (m * m)) / (2 * nrows)

  gradient _ x y theta = ((LA.tr x) #> (x #> theta - y)) / nrows
    where nrows = fromIntegral $ LAD.rows x
