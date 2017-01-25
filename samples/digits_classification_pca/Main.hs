module Main where

import qualified Numeric.LinearAlgebra as LA
import qualified MachineLearning.Types as T
import qualified MachineLearning as ML
import qualified MachineLearning.Classification.OneVsAll as OVA
import qualified MachineLearning.PCA as PCA


processFeatures :: T.Matrix -> T.Matrix
processFeatures = ML.addBiasDimension . (ML.mapFeatures 2)


calcAccuracy :: T.Matrix -> T.Vector -> [T.Vector] -> Double
calcAccuracy x y thetas = OVA.calcAccuracy y yPredicted
  where yPredicted = OVA.predict x thetas


main = do
  -- Step 1. Data loading.
  -- Step 1.1 Training Data loading.
  (x, y) <- pure ML.splitToXY <*> LA.loadMatrix "samples/digits_classification/optdigits.tra"
  -- Step 1.1 Testing Data loading.
  (xTest, yTest) <- pure ML.splitToXY <*> LA.loadMatrix "samples/digits_classification/optdigits.tes"

  -- Step 2. Outputs and features preprocessing.
  let numLabels = 10
      x' = processFeatures x
  -- Step 3. Dimensionality Reduction using PCA.
      (reduceDims, reducedDimensions, x1) = PCA.getDimReducer_rv x' 0.99999
      initialTheta = LA.konst 0 (LA.cols x1)
      initialThetas = replicate numLabels initialTheta
  -- Step 4. Learning.
      (thetas, optPath) = OVA.learn (OVA.BFGS2 0.1 0.5) 0.0001 30 30 numLabels x1 y initialThetas
  -- Step 5. Prediction and checking accuracy
      accuracyTrain = calcAccuracy x1 y thetas
      accuracyTest = calcAccuracy (reduceDims $ processFeatures xTest) yTest thetas

  -- Step 6. Printing results.
  putStrLn "\n=== Logistic Regression with PCA (Digits Recognition) ==="

  putStrLn $ "\nNumber of iterations to learn: " ++ show (map LA.rows optPath)
  

  putStrLn $ "\nReduced from " ++ show (LA.cols x') ++ " to " ++ show (LA.cols x1) ++ " dimensions"

  putStrLn $ "\nAccuracy on train set (%): " ++ show (accuracyTrain*100)
  putStrLn $ "Accuracy on test set (%): " ++ show (accuracyTest*100)
