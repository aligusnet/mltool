{-|
Module: MachineLearning.Regression.LeastSquares
Description: Least Squares
Copyright: (c) Alexander Ignatyev, 2016
License: BSD-3
Stability: experimental
Portability: POSIX

-}

module MachineLearning.Regression.LeastSquares
(
  LeastSquaresModel(..)
)

where

import qualified Numeric.LinearAlgebra as LA
import Numeric.LinearAlgebra((<>), (#>), (<.>))
import qualified Numeric.LinearAlgebra.Data as LAD
import qualified Data.Vector.Storable as V

import MachineLearning.Regression.Model

data LeastSquaresModel = LeastSquares

instance Model LeastSquaresModel where
  hypothesis _ x theta = x #> theta

  cost _ lambda x y theta = 
    let m = x #> theta - y
        nFeatures = V.length theta
        nExamples = fromIntegral $ LA.rows x
        thetaReg = V.slice 1 (nFeatures-1) theta
        regTerm = (thetaReg <.> thetaReg) * lambda
    in (LA.sumElements (m * m) + regTerm) / (2 * nExamples)

  gradient _ lambda x y theta = ((LA.tr x) #> (x #> theta - y) + regTerm) / nExamples
    where nExamples = fromIntegral $ LAD.rows x
          thetaReg = theta V.// [(0, 0)]
          regTerm = (LA.scalar lambda) * thetaReg
