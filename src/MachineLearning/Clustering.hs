{-|
Module: MachineLearning.Clustering
Description: Clustering
Copyright: (c) Alexander Ignatyev, 2017
License: BSD-3
Stability: experimental
Portability: POSIX

Cluster Analysis a.k.a. Clustering - grouping data into coherent subsets.
-}

module MachineLearning.Clustering
(
  Cluster(..)
  , kmeans
)

where

import Types (R, Vector, Matrix)
import Data.List (sortOn, groupBy, minimumBy)
import Control.Applicative ((<$>))
import Control.Monad (forM)
import qualified System.Random as Rnd
import qualified Control.Monad.Random as RndM
import qualified Data.Vector as V
import qualified Numeric.LinearAlgebra as LA
import MachineLearning.Random (sampleM)


-- | Cluster type (list of samples associtaed with the cluster).
type Cluster = V.Vector Vector


-- | Gets list if the nearest centroid to the sample.
nearestCentroidIndex :: V.Vector Vector  -- ^ list of cluster cetroids;
                     -> Vector            -- ^ sample;
                     -> Int               -- ^ index of the nearest centroid.
nearestCentroidIndex centroids v =
  let distances = V.map (\centroid -> LA.norm_2 (v-centroid)) centroids
  in V.minIndex distances


-- | Calculates cost associated with one claster.
calcClusterCost :: Cluster  -- ^ cluster;
                -> Vector   -- ^ cluster centroid;
                -> R        -- ^ cost value.
calcClusterCost cluster centroid = sum $ fmap (\sample -> LA.norm_2 $ sample-centroid) cluster


-- | Calculates cost value for all clusters.
calcCost :: V.Vector Cluster  -- ^ cluster list;
         -> V.Vector Vector   -- ^ list of cluster centroids;
         -> R                  -- ^ cost value.
calcCost clusters centroids = sum $ V.zipWith calcClusterCost clusters centroids


-- | Calculates centroid (middle point) of the given cluster.
getNewCentroid :: Cluster      -- ^ cluster;
               -> Vector       -- ^ centroid.
getNewCentroid cluster =
  let n = length cluster
      centroid = (sum cluster) / (fromIntegral n)
  in centroid


-- | Calculates new cluster centroids for each cluster.
moveCentroids :: V.Vector Cluster    -- ^ list of clusters;
              -> V.Vector Vector     -- ^ list of cluster centroids.
moveCentroids clusters = fmap getNewCentroid clusters


-- | Build cluster list from list of clusters indices.
buildClusterList :: V.Vector Vector   -- ^ list of samples;
                 -> V.Vector Int      -- ^ list of cluster indices (associated cluster index for each sample);
                 -> V.Vector Cluster  -- ^ list of clusters.
buildClusterList samples clusterIndicesList = V.fromList $ fmap getClusterSamples clusters''
  where clusters' = groupBy (\l r -> snd l == snd r) $ sortOn snd $ zip [0..] $ V.toList clusterIndicesList
        clusters'' = map (map fst) clusters'
        getClusterSamples clusterIndices = V.fromList $ fmap (samples V.!) clusterIndices


-- -- | Run K-Means algorithm once.
kmeansIter :: V.Vector Vector           -- ^ list of samples;
              -> Int                    -- ^ number of clusters (`K`);
              -> V.Vector Vector        -- ^ list of initial centroids;
              -> (V.Vector Cluster, R)  -- ^ (list of clusters, cost value).
kmeansIter samples k initialCentroids =
  let iter centroids =
        let clusterIndicesList = fmap (nearestCentroidIndex centroids) samples
            clusters = buildClusterList samples clusterIndicesList
            centroids' = moveCentroids clusters
            j = calcCost clusters centroids'
            diff = sum . fmap LA.norm_2 $ V.zipWith (-) centroids centroids'
        in if diff < 0.001 then (clusters, j)
           else iter centroids'
  in iter initialCentroids


-- | Run K-Means algorithm once inside Random Monad.
kmeansIterM :: Rnd.RandomGen g =>
               V.Vector Vector  -- ^ list of samples;
               -> Int           -- ^ number of clusters (`K`);
               -> Int           -- ^ iteration number;
               -> RndM.Rand g (V.Vector Cluster, R)  -- ^ (list of clusters, cost value) inside Random Monad.
kmeansIterM samples k _ = do
  centroids <- sampleM k samples
  return (kmeansIter samples k centroids)


-- | Clusters data using K-Means Algorithm.
-- Runs K-Means algorithm `N` times, returns the clustering with smaller cost.
kmeans :: Rnd.RandomGen g =>
          Int                         -- ^ number of K-Means Algorithm runs (`N`);
          -> Matrix                   -- ^ data to cluster;
          -> Int                      -- ^ desired number of clusters (`K`);
          -> g                        -- ^ randome generator;
          -> (V.Vector Cluster, g)    -- ^ list of clusters.
kmeans nIters x k gen = RndM.runRand (kmeansM nIters x k) gen


-- | Clusters data using K-Means Algorithm inside Random Monad.
-- Runs K-Means algorithm `N` times, returns the clustering with smaller cost.
kmeansM :: Rnd.RandomGen g =>
           Int ->                     -- ^ number of K-Means Algorithm runs (`N`);
           Matrix ->                  -- ^ data to cluster;
           Int ->                     -- ^ desired number of clusters (`K`);
           RndM.Rand g (V.Vector Cluster)  -- ^ list of clusters inside Random Monad.
kmeansM nIters x k = fst <$>
    (minimumBy (\(_, j1) (_, j2) -> compare j1 j2)) <$>
    forM [1..nIters] (kmeansIterM samples k)
  where samples = V.fromList $ LA.toRows x
