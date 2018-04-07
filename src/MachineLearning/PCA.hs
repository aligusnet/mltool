{-|
Module: MachineLearning.PCA
Description: Principal Component Analysis.
Copyright: (c) Alexander Ignatyev, 2017-2018.
License: BSD-3
Stability: experimental
Portability: POSIX

Principal Component Analysis (PCA) - dimensionality reduction algorithm.
It is mostly used to speed up supervising learning (Regression, Classification, etc) and visualization of data.
-}

module MachineLearning.PCA
(
  getDimReducer
  , getDimReducer_rv
)

where

import Prelude hiding ((<>))
import Data.Maybe (fromMaybe)
import qualified Data.Vector.Storable as V
import qualified Numeric.LinearAlgebra as LA
import Numeric.LinearAlgebra ((<>), (??))
import qualified MachineLearning as ML

import MachineLearning.Types (R, Vector, Matrix)


-- | Computes "covariance matrix".
covarianceMatrix :: Matrix -> Matrix
covarianceMatrix x = ((LA.tr x) <> x) / (fromIntegral $ LA.rows x)


-- | Compute eigenvectors (matrix U) and singular values (matrix S) of the given covariance matrix.
pca :: Matrix -> (Matrix, Vector)
pca x = (u, s)
  where sigma = covarianceMatrix x
        (u, s, _) = LA.svd sigma


-- | Gets dimensionality reduction function, retained variance (0..1) and reduced X
-- for given matrix X and number of dimensions to retain.
getDimReducer :: Matrix -> Int -> (Matrix -> Matrix, R, Matrix)
getDimReducer x k = (reducer, retainedVariance, reducer xNorm)
  where muSigma = ML.meanStddev x
        xNorm = ML.featureNormalization muSigma x
        (u, s) = pca xNorm
        u' = u ?? (LA.All, LA.Take k)
        reducer xx = (ML.featureNormalization muSigma xx) <> u'
        retainedVariance = (V.sum $ V.slice 0 k s) / (V.sum s)


-- | Gets dimensionality reduction function, retained number of dimensions and reduced X
-- for given matrix X and variance to retain (0..1].
getDimReducer_rv :: Matrix -> R -> (Matrix -> Matrix, Int, Matrix)
getDimReducer_rv x rv = (reducer, k, reducer xNorm)
  where muSigma = ML.meanStddev x
        xNorm = ML.featureNormalization muSigma x
        (u, s) = pca xNorm
        sum_s = V.sum s
        variances = (V.scanl' (+) 0 s) / (LA.scalar sum_s)
        k = fromMaybe ((V.length s) - 1) $ V.findIndex (\v -> v >= rv) variances
        u' = u ?? (LA.All, LA.Take k)
        reducer xx = (ML.featureNormalization muSigma xx) <> u'
