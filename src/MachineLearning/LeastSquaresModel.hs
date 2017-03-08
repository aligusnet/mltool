{-|
Module: MachineLearning.LeastSquaresModel
Description: Least Squares Model
Copyright: (c) Alexander Ignatyev, 2016
License: BSD-3
Stability: experimental
Portability: POSIX

-}

module MachineLearning.LeastSquaresModel
(
  LeastSquaresModel(..)
)

where

import qualified Numeric.LinearAlgebra as LA
import Numeric.LinearAlgebra((<>), (#>), (<.>))
import qualified Numeric.LinearAlgebra.Data as LAD
import qualified Data.Vector.Storable as V

import qualified MachineLearning.Regularization as R

import MachineLearning.Model

data LeastSquaresModel = LeastSquares

instance Model LeastSquaresModel where
  hypothesis LeastSquares x theta = x #> theta

  cost LeastSquares lambda x y theta = 
    let m = x #> theta - y
        nExamples = fromIntegral $ LA.rows x
        regTerm = R.costReg lambda theta
    in (LA.sumElements (m * m) * 0.5 + regTerm) / nExamples

  gradient LeastSquares lambda x y theta = ((LA.tr x) #> (x #> theta - y) + regTerm) / nExamples
    where nExamples = fromIntegral $ LAD.rows x
          regTerm = R.gradientReg lambda theta
