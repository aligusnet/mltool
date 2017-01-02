module Main where

import qualified Numeric.LinearAlgebra as LA
import qualified Types as T
import qualified MachineLearning as ML
import qualified MachineLearning.Classification as MLC
import qualified MachineLearning.PCA as PCA
import qualified MachineLearning.Regression.Logistic as LR

processFeatures :: T.Matrix -> T.Matrix
processFeatures = ML.addColumnOfOnes . (ML.mapFeatures 2)

calcAccuracy :: T.Matrix -> T.Vector -> [T.Vector] -> Double
calcAccuracy x y thetas = MLC.calcAccuracy y yPredicted
  where yPredicted = MLC.predictMulti x thetas

printOptPath x optPath =
  let thetas = optPath LA.?? (LA.All, LA.Drop 2)
      thetas_norm = LA.col $ map (LA.norm_2) $ LA.toRows . log $ 1 - LR.sigmoid (thetas LA.<> LA.tr x)
  in LA.disp 3 $ (optPath LA.?? (LA.All, LA.Take 2)) LA.||| thetas_norm

printInfinities x thetaList =
  let thetas = LA.fromColumns thetaList
      y :: T.Matrix
      y = log $ 1-LR.sigmoid (x LA.<> thetas)
      inf = 1/0
      xList = LA.toRows x
      x' = LA.fromRows $ map (\(xi, _) -> xList !! xi) $ LA.find (<=(-inf)) y
      z = x' LA.<> thetas
      h = LR.sigmoid z
      h' = 1 - h
      logh = log h
      logh' = log h'
  in do
    putStrLn "X"
    LA.disp 3 x'
    putStrLn "Z = X * Theta"
    LA.disp 3 z
    putStrLn "h(Theta) = sigmoid(Z)"
    LA.disp 3 h
    putStrLn "1 - h(Theta)"
    LA.disp 3 h'
    putStrLn "log (h(Theta))"
    LA.disp 3 logh
    putStrLn "log (1-h(Theta))"
    LA.disp 3 logh'

main = do
  -- Step 1. Data loading.
  -- Step 1.1 Training Data loading.
  (x, y) <- pure ML.splitToXY <*> LA.loadMatrix "samples/digits_classification/optdigits.tra"
  -- Step 1.1 Testing Data loading.
  (xTest, yTest) <- pure ML.splitToXY <*> LA.loadMatrix "samples/digits_classification/optdigits.tes"

  -- Step 2. Outputs and features preprocessing.
  let ys = MLC.processOutputMulti 10 y
      x' = processFeatures x
      (reduceDims, reducedDimensions, x1) = PCA.getDimReducer x' 10
      initialTheta = LA.konst 0 (LA.cols x1)
  -- Step 3. Learning.
      (thetaList, optPath) = MLC.learnMulti (MLC.BFGS2 0.1 0.5) 0.0001 30 30 x1 ys initialTheta
  -- Step 4. Prediction and checking accuracy
      accuracyTrain = calcAccuracy x1 y thetaList
      accuracyTest = calcAccuracy (reduceDims $ processFeatures xTest) yTest thetaList

  -- Step 5. Printing results.
  putStrLn "\n=== Numerical Issues Demonstration ==="

  putStrLn $ "\nNumber of iterations to learn: " ++ show (map LA.rows optPath)

  putStrLn $ "\nReduced from " ++ show (LA.cols x') ++ " to " ++ show (LA.cols x1) ++ " dimensions"

  putStrLn "\nOptimization Paths (# iter, J(Theta), norm_2(Theta))"
  mapM_ (printOptPath x1) optPath

  print "\nDisplay numerical issues"
  printInfinities x1 thetaList

  putStrLn $ "\nAccuracy on train set (%): " ++ show (accuracyTrain*100)
  putStrLn $ "Accuracy on test set (%): " ++ show (accuracyTest*100)
