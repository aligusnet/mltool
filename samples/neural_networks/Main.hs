module Main where

import qualified Numeric.LinearAlgebra as LA
import qualified MachineLearning as ML
import qualified MachineLearning.Optimization as Opt
import qualified MachineLearning.NeuralNetwork as NN
import qualified MachineLearning.TerminalProgress as TP

main = do
  putStrLn "\n== Neural Networks (Digits Recognition) ==\n"

  -- Step 1. Data loading.
  -- Step 1.1 Training Data loading.
  (x, y) <- pure ML.splitToXY <*> LA.loadMatrix "samples/digits_classification/optdigits.tra"
  -- Step 1.1 Testing Data loading.
  (xTest, yTest) <- pure ML.splitToXY <*> LA.loadMatrix "samples/digits_classification/optdigits.tes"

  -- Step 2. Initialize Neural Network.
  let nnt = NN.makeTopology (LA.cols x) 10 [100, 100]
      model = NN.NeuralNetwork nnt
  -- Step 3. Initialize theta with randon values.
      initTheta = NN.initializeTheta 5191711 nnt

      x1 = ML.addColumnOfOnes x

  -- Step 4. Learn the Neural Network.
  (thetaNN, optPath) <- TP.learnWithProgressBar (Opt.minimize (Opt.BFGS2 0.03 0.7) model 1e-7 5 5 x1 y) initTheta 20

  -- Step 5. Making predictions and checking accuracy on training and test sets.
  let accuracyTrain = NN.calcAccuracy y (NN.hypothesis model x1 thetaNN)
      accuracyTest = NN.calcAccuracy yTest (NN.hypothesis model (ML.addColumnOfOnes xTest) thetaNN)

  -- Step 6. Printing results.

  putStrLn $ "\nNumber of iterations to learn the Neural Network: " ++ show (LA.rows optPath)

  putStrLn $ "\nAccuracy on train set (%): " ++ show (accuracyTrain*100)
  putStrLn $ "Accuracy on test set (%): " ++ show (accuracyTest*100)
