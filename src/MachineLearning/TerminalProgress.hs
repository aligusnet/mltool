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
)

where

import Types (Vector, Matrix)
import Control.DeepSeq (deepseq)
import qualified System.Console.AsciiProgress as AP
import qualified Numeric.LinearAlgebra as LA


-- | Learn the given function displaying progress bar in terminal.
-- It takes function, initial theta and number of iterations to call the function.
-- It returns theta and optimization path (see "MachineLearning.Regression" for details).
learnWithProgressBar :: (Vector -> (Vector, Matrix)) -> Vector -> Integer -> IO (Vector, Matrix)
learnWithProgressBar func initialTheta nIterations = AP.displayConsoleRegions $ do
  pg <- AP.newProgressBar AP.def {
    AP.pgTotal = nIterations-1 -- somehow prograress bar does 1 iteration more
    , AP.pgFormat = "Learning :percent [:bar] (for :elapsed, :eta remaining)"
    }
  loop pg initialTheta []
    where loop pg theta optPaths = do
            b <- AP.isComplete pg
            if b
              then return (theta, buildOptPathMatrix $ reverse optPaths)
              else do
              let (theta', optPath) = func theta
              (theta', optPath) `deepseq` AP.tick pg
              loop pg theta' (optPath : optPaths)              


-- | Build a single optimazation path matrix from list of optimization path matrices.
buildOptPathMatrix :: [Matrix] -> Matrix
buildOptPathMatrix matrices = LA.fromBlocks $ map (\m -> [m]) matrices
