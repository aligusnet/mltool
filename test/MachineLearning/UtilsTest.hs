module MachineLearning.UtilsTest
(
  tests
)

where

import Test.Framework (testGroup)
import Test.Framework.Providers.HUnit
import Test.HUnit
import Test.HUnit.Approx
import Test.HUnit.Plus

import MachineLearning.Types (Vector, Matrix)
import qualified Numeric.LinearAlgebra as LA
import qualified Data.Vector.Storable as V
import Numeric.LinearAlgebra ((><))

import MachineLearning.Utils

a :: Matrix
a = (4><5) [ 1, 7, 9, 11, 2
           , 77, 4, 6, 9, 0
           , -11, -55, 3, 11, 55
           , 7, 9, 11, 13, 15
           ]

sumRows :: Matrix
sumRows = (4><1) [ 30
                 , 96
                 , 3
                 , 55
                 ]


sumCols :: Matrix
sumCols = (1><5) [74, -35, 29, 44, 72]


maxRows :: Matrix
maxRows = (4><1) [ 11
                 , 77
                 , 55
                 , 15
                 ]


minCols :: Matrix
minCols = (1><5) [-11, -55, 3, 9, 0]


sumRowsV = LA.flatten sumRows
sumColsV = LA.flatten sumCols
maxRowsV = LA.flatten maxRows
minColsV = LA.flatten minCols


tests = [ testGroup "reduceV" [
            testCase "reduceByRowsV: sum" $ assertVector "" 1e-10 sumRowsV (reduceByRowsV V.sum a)
            , testCase "reduceByColumnsV: sum" $ assertVector "" 1e-10 sumColsV (reduceByColumnsV V.sum a)
            , testCase "reduceByRowsV: max" $ assertVector "" 1e-10 maxRowsV (reduceByRowsV V.maximum a)
            , testCase "reduceByColumnsV: min" $ assertVector "" 1e-10 minColsV (reduceByColumnsV V.minimum a)
            ]
          , testGroup "reduce" [
              testCase "reduceByRows: sum" $ assertMatrix "" 1e-10 sumRows (reduceByRows V.sum a)
              , testCase "reduceByColumns: sum" $ assertMatrix "" 1e-10 sumCols (reduceByColumns V.sum a)
              , testCase "reduceByRows: max" $ assertMatrix "" 1e-10 maxRows (reduceByRows V.maximum a)
              , testCase "reduceByColumns: min" $ assertMatrix "" 1e-10 minCols (reduceByColumns V.minimum a)
            ]
          , testGroup "sum" [
              testCase "sumByRows" $ assertMatrix "" 1e-10 sumRows (sumByRows a)
              , testCase "sumColumns" $ assertMatrix "" 1e-10 sumCols (sumByColumns a)
              ]
          , testCase "listOfTuplesToList" $ assertEqual "" [11, 9, 27, 3, 43, 11] (listOfTuplesToList [(11, 9), (27, 3), (43, 11)])
        ]
