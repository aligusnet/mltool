module Main where

import qualified Numeric.LinearAlgebra as LA
import qualified MachineLearning.Types as T
import qualified MachineLearning as ML
import qualified MachineLearning.Classification.Binary as CB
import qualified MachineLearning.Classification.OneVsAll as OVA
import qualified MachineLearning.TerminalProgress as TP
import qualified MachineLearning.Optimization as Opt
import MachineLearning.MultiSvmClassifier
import MachineLearning.SoftmaxClassifier


featuresMapParameter = 2

processFeatures :: T.Matrix -> T.Matrix
processFeatures muSigma = ML.addBiasDimension . ML.featureNormalization muSigma . ML.mapFeatures featuresMapParameter

calcAccuracy :: (Model a) => a -> T.Matrix -> T.Vector -> T.Vector -> Double
calcAccuracy m x y theta = OVA.calcAccuracy y yPredicted
  where yPredicted = hypothesis m x theta

main = do
  putStrLn "\n=== Optical Recognition of Handwritten Digits Data Set ===\n"
  -- Step 1. Data loading.
  -- Step 1.1 Training Data loading.
  (x, y) <- pure ML.splitToXY <*> LA.loadMatrix "digits_classification/optdigits.tra"
  -- Step 1.1 Testing Data loading.
  (xTest, yTest) <- pure ML.splitToXY <*> LA.loadMatrix "digits_classification/optdigits.tes"

  -- Step 2. Outputs and features preprocessing.
  let numLabels = 10
      svm = MultiClass $ MultiSvm 1 numLabels
      softmax = MultiClass $ Softmax numLabels
      muSigma = ML.meanStddev (ML.mapFeatures featuresMapParameter x)
      x1 = processFeatures muSigma x
      xTest1 = processFeatures muSigma xTest
      initialTheta = LA.konst 0.1 (numLabels*(LA.cols x1))
  print $ LA.size x1
  print $ LA.size $ x1 LA.? [1..10]
  -- Step 3. Learning.
  putStrLn "Learning Multi SVM model..."
  (thetaSvm, optPathSvm) <- TP.learnWithProgressBar (Opt.minimize (Opt.BFGS2 0.01 0.7) svm 0.0001 5 (CB.L2 2) x1 y) initialTheta 20

  putStrLn "\nLearning MGD Softmax model"
  --(thetaSm, optPathSm) <- TP.learnWithProgressBar (Opt.minimize (Opt.BFGS2 0.01 0.7) softmax 0.000001 5 1 x1 y) initialTheta 20
  (thetaSm, optPathSm) <- TP.learnWithProgressBar (Opt.minimize (Opt.MinibatchGradientDescent 0 1024 0.01) svm 0.000001 200 (CB.L2 1) x1 y) initialTheta 20

  putStrLn "\nLearning GD Softmax model"
  (thetaSmGD, optPathSmGD) <- TP.learnWithProgressBar (Opt.minimize (Opt.GradientDescent 0.01) softmax 0.000001 50 (CB.L2 1) x1 y) initialTheta 20

  -- Step 4. Prediction and checking accuracy
  let accuracyTrainSvm = calcAccuracy svm x1 y thetaSvm
      accuracyTestSvm = calcAccuracy svm xTest1 yTest thetaSvm

  -- Step 5. Printing results.
  putStrLn $ "\nSVM: Number of iterations to learn: " ++ show (LA.rows optPathSvm)

  putStrLn $ "SVM: Accuracy on train set (%): " ++ show (accuracyTrainSvm*100)
  putStrLn $ "SVM: Accuracy on test set (%): " ++ show (accuracyTestSvm*100)

  let accuracyTrainSm = calcAccuracy softmax x1 y thetaSm
      accuracyTestSm = calcAccuracy softmax xTest1 yTest thetaSm

  -- Step 5. Printing results.
  putStrLn $ "\nSoftmax: Number of iterations to learn: " ++ show (LA.rows optPathSm)

  putStrLn $ "Softmax: Accuracy on train set (%): " ++ show (accuracyTrainSm*100)
  putStrLn $ "Softmax: Accuracy on test set (%): " ++ show (accuracyTestSm*100)

  let r = LA.rows optPathSm
  LA.dispShort 150 8 8 $ optPathSm LA.? [r-151 .. r-1]

  let accuracyTrainSmGD = calcAccuracy softmax x1 y thetaSmGD
      accuracyTestSmGD = calcAccuracy softmax xTest1 yTest thetaSmGD

  -- Step 5. Printing results.
  putStrLn $ "\nGD Softmax: Number of iterations to learn: " ++ show (LA.rows optPathSmGD)

  putStrLn $ "GD Softmax: Accuracy on train set (%): " ++ show (accuracyTrainSmGD*100)
  putStrLn $ "GD Softmax: Accuracy on test set (%): " ++ show (accuracyTestSmGD*100)

