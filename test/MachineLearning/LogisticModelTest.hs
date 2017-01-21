module MachineLearning.LogisticModelTest
(
  tests
)

where

import Test.Framework (testGroup)
import Test.Framework.Providers.HUnit
import Test.HUnit
import Test.HUnit.Approx
import Test.HUnit.Plus

import MachineLearning.DataSets (dataset2)

import qualified Numeric.LinearAlgebra as LA
import qualified MachineLearning as ML
import MachineLearning.Optimization (checkGradient)
import MachineLearning.Model
import MachineLearning.LogisticModel

(x, y) = ML.splitToXY dataset2

x1 = ML.addBiasDimension $ ML.mapFeatures 6 x
onesTheta :: LA.Vector LA.R
onesTheta = LA.konst 1 (LA.cols x1)
zeroTheta :: LA.Vector LA.R
zeroTheta = LA.konst 0 (LA.cols x1)

gradientCheckingEps = 1e-3

tests = [ testGroup "sigmoid" [
            testCase "zero" $ assertApproxEqual "" 1e-10 0.5 (sigmoid 0)
            , testCase "big positive value" $ assertApproxEqual "" 1e-10 1 (sigmoid 10e10)
            , testCase "big negative value" $ assertApproxEqual "" 1e-10 0 (sigmoid $ -10e10) 
            , testCase "quite big positive value" $ assertApproxEqual "" 1e-2 1 (sigmoid 100)
            , testCase "quite big negative value" $ assertApproxEqual "" 1e-2 0 (sigmoid $ -100)
            ]
          , testGroup "sigmoidGradient" [
              testCase "zero" $ assertApproxEqual "" 1e-10 0.25 (sigmoidGradient 0)
              , testCase "big positive value" $ assertApproxEqual "" 1e-10 0 (sigmoidGradient 10e10)
              , testCase "big negative value" $ assertApproxEqual "" 1e-10 0 (sigmoidGradient $ -10e10) 
              , testCase "quite big positive value" $ assertApproxEqual "" 1e-2 0 (sigmoidGradient 100)
              , testCase "quite big negative value" $ assertApproxEqual "" 1e-2 0 (sigmoidGradient $ -100)
              , testCase "small positive value" $ assertApproxEqual "" 1e-2 0.2 (sigmoidGradient 1)
              , testCase "small negative value" $ assertApproxEqual "" 1e-2 0.2 (sigmoidGradient $ -1) 
              ]
          , testGroup "model" [
              testCase "cost, lambda = 0" $ assertApproxEqual "" 1e-3 2.020 (cost Logistic 0 x1 y onesTheta)
              , testCase "cost, lambda = 1" $ assertApproxEqual "" 1e-3 2.135 (cost Logistic 1 x1 y onesTheta)
              , testCase "cost, lambda = 1000" $ assertApproxEqual "" 1e-3 116.427 (cost Logistic 1000 x1 y onesTheta)
              , testCase "gradient, lambda = 0" $ assertVector "" 1e-5 gradient_l0 (gradient Logistic 0 x1 y onesTheta)
              , testCase "gradient, lambda = 1" $ assertVector "" 1e-5 gradient_l1 (gradient Logistic 1 x1 y onesTheta)
              , testCase "gradient, lambda = 1000" $ assertVector "" 1e-5 gradient_l1000 (gradient Logistic 1000 x1 y onesTheta)
              ]
          , testGroup "gradient checking" [
              testCase "non-zero theta, non-zero lambda" $ assertApproxEqual "" gradientCheckingEps 0 (checkGradient Logistic 2 x1 y onesTheta 1e-3)
              , testCase "zero theta, non-zero lambda" $ assertApproxEqual "" gradientCheckingEps 0 (checkGradient Logistic 2 x1 y zeroTheta 1e-3)
              , testCase "non-zero theta, zero lambda" $ assertApproxEqual "" gradientCheckingEps 0 (checkGradient Logistic 0 x1 y onesTheta 1e-3)
              , testCase "zero theta, zero lambda" $ assertApproxEqual "" gradientCheckingEps 0 (checkGradient Logistic 0 x1 y zeroTheta 1e-3)
              ]
        ]

gradient_l0 = LA.vector [0.34604507367924525,7.660615656904722e-2,0.11004999290013262,0.14211701951318526,7.4399123914273965e-3,0.15963981400206023,5.864636026438637e-2,2.369595228045015e-2,1.7568631688383615e-2,9.87226947583087e-2,8.878427090266403e-2,2.509755728155993e-3,3.348199373717313e-2,1.0975419586624715e-3,0.11520318246520422,5.0480764463147594e-2,1.0229511332420786e-2,8.818652179760128e-3,1.5052078581608905e-2,6.655810974747264e-3,9.010665405440292e-2,6.480865498500844e-2,2.039892642614106e-3,1.4231095091583643e-2,5.737477462013599e-4,1.71609000731311e-2,-2.437841009425525e-4,9.753746346629336e-2]

gradient_l1 = LA.vector [0.34604507367924525,8.508073284023365e-2,0.11852456917131905,0.15059159578437167,1.5914488662613836e-2,0.16811439027324665,6.71209365355728e-2,3.217052855163659e-2,2.6043207959570054e-2,0.10719727102949513,9.725884717385046e-2,1.0984331999342433e-2,4.1956570008359576e-2,9.572118229848912e-3,0.12367775873639067,5.895534073433403e-2,1.8704087603607224e-2,1.729322845094657e-2,2.3526654852795346e-2,1.5130387245933704e-2,9.858123032558937e-2,7.328323125619488e-2,1.0514468913800546e-2,2.270567136277008e-2,9.048324017387801e-3,2.563547634431754e-2,8.230792170243889e-3,0.10601203973747979]

gradient_l1000 = LA.vector [0.34604507367924525,8.551182427755489,8.584626264086573,8.616693290699626,8.48201618357787,8.6342160851885,8.533222631450828,8.498272223466891,8.492144902874823,8.57329896594475,8.563360542089104,8.477086026914597,8.508058264923614,8.475673813145104,8.589779453651644,8.525057035649588,8.484805782518862,8.483394923366202,8.489628349768049,8.481232082161188,8.564682925240843,8.539384926171449,8.476616163829055,8.488807366278024,8.475150018932643,8.491737171259572,8.474332487085498,8.572113734652733]
