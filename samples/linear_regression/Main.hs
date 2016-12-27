module Main where

import qualified Numeric.LinearAlgebra as LA
import qualified MachineLearning as ML
import qualified MachineLearning.Regression as MLR

main = do
  -- Step 1. Data loading
  m <- LA.loadMatrix "samples/linear_regression/data.txt"
  let (x, y) = ML.splitToXY m

  -- Step 2. Feature Normalization
      muSigma = ML.meanStddev x
      xNorm = ML.featureNormalization muSigma x
      x1 = ML.addColumnOfOnes xNorm

  -- Step 3. Learning using 3 methods:
  -- NE: Normal Equation (exact solution)
  -- GD: Gradient Descent (basic iterative method)
  -- BFGS: BFGS (most advanced iterative method)
      zeroTheta = LA.konst 0 (LA.cols x1)
      thetaNE = ML.normalEquation x1 y
      (thetaGD, optPathGD) = MLR.minimize (MLR.GradientDescent 0.01) MLR.LeastSquares 0.0001 5000 0 x1 y zeroTheta
      (thetaBFGS, optPathBFGS) = MLR.minimize (MLR.BFGS2 0.1 0.1)    MLR.LeastSquares 0.0001 1500 0 x1 y zeroTheta

  -- Step 4. Prediction
      xPredict = LA.matrix 2 [1650, 3]
      xPredict1 = ML.addColumnOfOnes $ ML.featureNormalization muSigma xPredict
      yPredictNE = MLR.hypothesis MLR.LeastSquares xPredict1 thetaNE
      yPredictGD = MLR.hypothesis MLR.LeastSquares xPredict1 thetaGD
      yPredictBFGS = MLR.hypothesis MLR.LeastSquares xPredict1 thetaBFGS

  -- Step 5. Printing results
  putStrLn "\n=== Linear Regression Sample Application ===\n"
  putStrLn "Feature normalization, mu:"
  LA.disp 4 $ fst muSigma
  putStrLn "Feature normalization, sigma:"
  LA.disp 4 $ snd muSigma
  putStrLn "Normilized features:"
  LA.dispShort 10 10 4 $ x LA.||| xNorm

  putStrLn ""
  putStrLn $ "Theta (Normal Equation):  " ++ (show thetaNE)
  putStrLn $ "Theta (Gradient Descent): " ++ (show thetaGD)
  putStrLn $ "Theta (BFGS):             " ++ (show thetaBFGS)

  putStrLn ""
  putStrLn $ "Prediction (Normal Equation):  " ++ (show yPredictNE)
  putStrLn $ "Prediction (Gradient Descent): " ++ (show yPredictGD)
  putStrLn $ "Prediction (BFGS):             " ++ (show yPredictBFGS)

  putStrLn ""
  putStrLn "Optimization Path (Gradient Descent):"
  LA.dispShort 20 10 4 optPathGD
  putStrLn "Optimization Path (BFGS2):"
  LA.disp 3 optPathBFGS
