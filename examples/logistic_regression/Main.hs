module Main where

import qualified MachineLearning.Types as T
import qualified Numeric.LinearAlgebra as LA
import qualified MachineLearning as ML
import qualified MachineLearning.Classification.Binary as BC


calcAccuracy :: T.Matrix -> T.Vector -> T.Vector -> T.R
calcAccuracy x y theta = BC.calcAccuracy y yPredicted
  where yPredicted = BC.predict x theta

main = do
  -- Step 1. Data loading.
  m <- LA.loadMatrix "logistic_regression/data.txt"
  let (x, y) = ML.splitToXY m
  -- Step 2. Feature normalization (skipped - we don't need feature normalization if we use BFGS2).
  -- See Linear Regression sample app for dedails.
  
  -- Step 3. Feature mapping.
      x1 = ML.addBiasDimension $ ML.mapFeatures 6 x

  -- Step 4. Learning
      zeroTheta = LA.konst 0 (LA.cols x1)
      (theta, _) = BC.learn (BC.BFGS2 0.1 0.1) 0.0001 1500 (BC.L2 1) x1 y zeroTheta

  -- Step 5. Prediction and checking accuracy
      accuracy = calcAccuracy x1 y theta

  -- Step 6. Printing results.
  putStrLn "\n=== Logistic Regression Sample Application ===\n"

  putStrLn ""
  putStrLn $ "Theta:             " ++ (show theta)
  
  putStrLn ""
  putStrLn $ "Accuracy on training set data (%): " ++ show (accuracy*100)
