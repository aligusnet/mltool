module Main where

import qualified Numeric.LinearAlgebra as LA
import qualified Types as T
import qualified MachineLearning as ML
import qualified MachineLearning.Classification as MLC

processFeatures :: T.Matrix -> T.Matrix
processFeatures = ML.addColumnOfOnes . (ML.mapFeatures 2)

calcAccuracy :: T.Matrix -> T.Vector -> [T.Vector] -> Double
calcAccuracy x y thetas = MLC.calcAccuracy y yPredicted
  where yPredicted = MLC.predictMulti x thetas

main = do
  -- Step 1. Data loading.
  -- Step 1.1 Training Data loading.
  (x, y) <- pure ML.splitToXY <*> LA.loadMatrix "samples/digits_classification/optdigits.tra"
  -- Step 1.1 Testing Data loading.
  (xTest, yTest) <- pure ML.splitToXY <*> LA.loadMatrix "samples/digits_classification/optdigits.tes"

  -- Step 2. Outputs and features preprocessing.
  let ys = MLC.processOutputMulti 10 y
      x1 = processFeatures x
      initialTheta = LA.konst 0 (LA.cols x1)
  -- Step 3. Learning.
      (thetas, _) = MLC.learnMulti (MLC.BFGS2 0.1 0.1) 0.001 30 30 x1 ys initialTheta
  -- Step 4. Prediction and checking accuracy
      accuracyTrain = calcAccuracy x1 y thetas
      accuracyTest = calcAccuracy (processFeatures xTest) yTest thetas

  -- Step 5. Printing results.
  putStrLn "\n=== Optical Recognition of Handwritten Digits Data Set ==="

  putStrLn $ "Accuracy on train set (%): " ++ show (accuracyTrain*100)
  putStrLn $ "Accuracy on test set (%): " ++ show (accuracyTest*100)
