module Main where

import qualified System.Random as Rnd
import qualified Numeric.LinearAlgebra as LA
import MachineLearning.Clustering
import qualified Data.Vector as V
      

main = do
  putStrLn "\n=== K-Means Sample Application ===\n"
  -- Step 1. Data loading.
  -- Step 1.1 Training Data loading.
  x <- LA.loadMatrix "samples/linear_regression/data.txt"
  gen <- Rnd.newStdGen
  -- Step 2. Learning
  let numIters = 25
      numClusters = 3
      (clusterList, _) = kmeans numIters x numClusters gen

  -- Step 3. Display results
  print $ "Successfully groupped into " ++ show (V.length clusterList) ++ " clusters"
