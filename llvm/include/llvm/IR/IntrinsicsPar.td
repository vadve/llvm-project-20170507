//===- IntrinsicsPar.td - Defines Parallelism Intrinsics ---*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines intrinsics used to describe OpenMP directives and pragmas
//
//===----------------------------------------------------------------------===//
                                      

//===------------------ Directive Region Intrinsics -----------------------===//
//
def int_directive_region_entry : Intrinsic<[llvm_token_ty], [], [], 
  "llvm.directive.region.entry">;
def int_directive_region_exit  : Intrinsic<[], [llvm_token_ty], [], 
  "llvm.directive.region.exit">;
def int_directive_marker : Intrinsic<[llvm_i1_ty], [], [], 
  "llvm.directive.marker">;

