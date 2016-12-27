module Main where

import qualified Types as T
import qualified Data.Vector.Storable as V
import qualified Numeric.LinearAlgebra as LA
import qualified MachineLearning as ML
import qualified MachineLearning.Regression as MLR


calcAccuracy :: T.Matrix -> T.Vector -> T.Vector -> T.R
calcAccuracy x y theta =
  let yPredict = V.map (\r -> if r >= 0.5 then 1 else 0) (MLR.hypothesis MLR.Logistic x theta)
      diff = abs(y - yPredict)
  in (1 - (V.sum diff) / (fromIntegral $ V.length diff)) * 100

main = do
  -- Step 1. Data loading.
  m <- LA.loadMatrix "samples/logistic_regression/data.txt"
  let (x, y) = ML.splitToXY m
  -- Step 2. Feature normalization (skipped - we don't need feature normalization if we use BFGS2).
  -- See Linear Regression sample app for dedails.
  
  -- Step 3. Feature mapping.
      x1 = ML.addColumnOfOnes $ ML.mapFeatures 6 x

  -- Step 4. Learning
      zeroTheta = LA.konst 0 (LA.cols x1)
      (thetaBFGS, optPathBFGS) = MLR.minimize (MLR.BFGS2 0.1 0.1) MLR.Logistic 0.0001 1500 1 x1 y zeroTheta

  -- Step 5. Prediction and checking accuracy
      accuracyBFGS = calcAccuracy x1 y thetaBFGS

  -- Step 6. Printing results.
  putStrLn "\n=== Logistic Regression Sample Application ===\n"

  putStrLn ""
  putStrLn $ "Theta (BFGS):             " ++ (show thetaBFGS)
  
  putStrLn ""
  putStrLn $ "Accuracy on training set data (BFGS): " ++ show accuracyBFGS
