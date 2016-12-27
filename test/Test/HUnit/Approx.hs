{-# LANGUAGE ImplicitParams, CPP #-}
#if __GLASGOW_HASKELL__ >= 707
{-# LANGUAGE Safe #-}       -- Test.HUnit is not Safe in 7.6 and below
#endif
-----------------------------------------------------------------------------
-- |
-- Module      :  Test.HUnit.Approx
-- Copyright   :  (C) 2014 Richard Eisenberg
-- License     :  BSD-style (see LICENSE)
-- Maintainer  :  Richard Eisenberg (eir@cis.upenn.edu)
-- Stability   :  intended to be stable
-- Portability :  not portable (uses implicit parameters)
--
-- This module exports combinators to allow approximate equality of
-- floating-point values in HUnit tests.
-----------------------------------------------------------------------------

module Test.HUnit.Approx (
  -- * Assertions
  assertApproxEqual, (@~?), (@?~),

  -- * Tests
  (~~?), (~?~)
  ) where

import Test.HUnit
import Control.Monad  ( unless )

-- | Asserts that the specified actual value is approximately equal to the
-- expected value. The output message will contain the prefix, the expected
-- value, the actual value, and the maximum margin of error.
--  
-- If the prefix is the empty string (i.e., @\"\"@), then the prefix is omitted
-- and only the expected and actual values are output.
assertApproxEqual :: (Ord a, Num a, Show a)
                  => String -- ^ The message prefix
                  -> a      -- ^ Maximum allowable margin of error
                  -> a      -- ^ The expected value 
                  -> a      -- ^ The actual value
                  -> Assertion
assertApproxEqual preface epsilon expected actual =
  unless (abs (actual - expected) <= epsilon) (assertFailure msg)
  where msg = (if null preface then "" else preface ++ "\n") ++
              "expected: " ++ show expected ++ "\n but got: " ++ show actual ++
              "\n (maximum margin of error: " ++ show epsilon ++ ")"

-- | Asserts that the specified actual value is approximately equal to the
-- expected value (with the expected value on the right-hand side). The margin
-- of error is specified with the implicit parameter @epsilon@.
(@?~) :: (Ord a, Num a, Show a, ?epsilon :: a)
      => a        -- ^ The actual value
      -> a        -- ^ The expected value
      -> Assertion
x @?~ y = assertApproxEqual "" ?epsilon y x
infix 1 @?~

-- | Asserts that the specified actual value is approximately equal to the
-- expected value (with the expected value on the left-hand side). The margin
-- of error is specified with the implicit parameter @epsilon@.
(@~?) :: (Ord a, Num a, Show a, ?epsilon :: a)
      => a     -- ^ The expected value
      -> a     -- ^ The actual value
      -> Assertion
x @~? y = assertApproxEqual "" ?epsilon x y
infix 1 @~?

-- | Shorthand for a test case that asserts approximate equality (with the
-- expected value on the left-hand side, and the actual value on the
-- right-hand side).
(~~?) :: (Ord a, Num a, Show a, ?epsilon :: a)
      => a     -- ^ The expected value
      -> a     -- ^ The actual value
      -> Test
expected ~~? actual = TestCase (expected @~? actual)
infix 1 ~~?

-- | Shorthand for a test case that asserts approximate equality (with the
-- actual value on the left-hand side, and the expected value on the
-- right-hand side).
(~?~) :: (Ord a, Num a, Show a, ?epsilon :: a)
      => a     -- ^ The actual value
      -> a     -- ^ The expected value 
      -> Test
actual ~?~ expected = TestCase (actual @?~ expected)
infix 1 ~?~
