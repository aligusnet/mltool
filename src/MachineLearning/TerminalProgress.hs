{-|
Module: TerminalProgress
Description: Learn function with progress bar for terminal.
Copyright: (c) Alexander Ignatyev, 2017
License: BSD-3
Stability: experimental
Portability: POSIX

Learn function with progress bar for terminal.
-}

module MachineLearning.TerminalProgress
(
  learnWithProgressBar
  , learnOneVsAllWithProgressBar
)

where

import Data.List (transpose)
import MachineLearning.Types (Vector, Matrix)
import qualified MachineLearning.Classification.Internal as MLC
import Control.Monad (foldM, mapAndUnzipM)
import Control.DeepSeq (deepseq)
import qualified System.Console.AsciiProgress as AP
import qualified Numeric.LinearAlgebra as LA


-- | Learn the given function displaying progress bar in terminal.
-- It takes function, initial theta and number of iterations to call the function.
-- It returns theta and optimization path (see "MachineLearning.Optimization" for details).
learnWithProgressBar :: (Vector -> (Vector, Matrix)) -> Vector -> Int -> IO (Vector, Matrix)
learnWithProgressBar func initialTheta nIterations = AP.displayConsoleRegions $ do
  pg <- newProgressBar nIterations
  (theta, optPaths) <- foldM (doLoop pg func) (initialTheta, []) [1..nIterations]
  return (theta, buildOptPathMatrix $ reverse optPaths)


-- | Learn the given function displaying progress bar in terminal.
-- It takes function, list of outputs and list of initial thetas and number of iterations to call the function.
-- It returns list of thetas and list of optimization paths (see "MachineLearning.Optimization" for details).
learnOneVsAllWithProgressBar :: (Vector -> Vector -> (Vector, Matrix)) -> Vector -> [Vector] -> Int -> IO ([Vector], [Matrix])
learnOneVsAllWithProgressBar func y initialThetaList nIterations = AP.displayConsoleRegions $ do
    let numLabels = length initialThetaList
        ys = MLC.processOutputOneVsAll numLabels y
    pg <- newProgressBar $ nIterations * (length ys)
    mapAndUnzipM (learnOneClass pg func nIterations) $ zip ys initialThetaList


newProgressBar nIterations = AP.newProgressBar AP.def {
  AP.pgTotal = fromIntegral nIterations
  , AP.pgFormat = "Learning :percent [:bar] (for :elapsed, :eta remaining)"
  }

doLoop pg func (theta, optPaths) _ = do
  let (theta', optPath) = func theta
  theta' `deepseq` AP.tick pg
  return (theta', (optPath : optPaths))


learnOneClass pg func nIterations (y, theta) = do
  (theta, optPaths) <- foldM (doLoop pg $ func y) (theta, []) [1..nIterations]
  return (theta, buildOptPathMatrix $ reverse optPaths)


-- | Build a single optimazation path matrix from list of optimization path matrices.
buildOptPathMatrix :: [Matrix] -> Matrix
buildOptPathMatrix matrices = LA.fromBlocks $ map (\m -> [m]) matrices
