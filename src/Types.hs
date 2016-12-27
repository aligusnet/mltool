{-|
Module: Types
Description: Common Types
Copyright: (c) Alexander Ignatyev, 2016
License: BSD-3
Stability: experimental
Portability: POSIX

Common type definitions used in all modules.
-}

module Types
(
    R
  , Vector
  , Matrix
)

where

import qualified Numeric.LinearAlgebra.Data as LAD

-- | Scalar Type (Double)
type R = LAD.R

-- | Vector Types (of Doubles)
type Vector = LAD.Vector R

-- | Matrix Types (of Doubles)
type Matrix = LAD.Matrix R

