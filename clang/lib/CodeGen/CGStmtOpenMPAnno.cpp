//=== CGStmtOpenMPAnno.cpp - Emit annotations from OMPExecutableDirective -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit the new codegen for OpenMP in LLVM
//
//===----------------------------------------------------------------------===//

//#include "CGCleanup.h"
#include "CGOpenMPRuntime.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/StmtOpenMP.h"
#include "clang/AST/DeclOpenMP.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace clang;
using namespace CodeGen;

// This is a workaround to use the class OMPLexicalScope without pulling it's
// definition out of CGStmtOpenMP.cpp.
// These two functions are wrapper functions defined in CGStmtOpenMP.cpp. the
// first dynamically constructs a OMPLexicalScope object and the second destroys
// it (and hence calls the destructor of OMPLexicalScope).
//
// In order to get the behaviour of OMPLexicalScope, an OMPLexicalScope2 object
// should be used
void* runOMPLexicalScope(CodeGenFunction &, const OMPExecutableDirective &);
void destroyOMPLexicalScope(void *p);

namespace {
typedef llvm::SmallVector<llvm::OperandBundleDef,8/*initialSize*/> BundleListTy;
typedef llvm::SmallPtrSet<const VarDecl*, 32> VarDeclSet32;
typedef llvm::SmallVector<OpenMPDirectiveKind, 4> OpenMPDirList;

class OMPLexicalScope2 {
  void *ptrToOMPLexicalScope;
public:
  OMPLexicalScope2(CodeGenFunction &CGF, const OMPExecutableDirective &S)
      : ptrToOMPLexicalScope(runOMPLexicalScope(CGF, S)) {}
  ~OMPLexicalScope2() {
    destroyOMPLexicalScope(ptrToOMPLexicalScope);
  }
};

enum OMPIfModifier {
  no_if,                // no IF clause specified
  if_none,              // if(x)
  if_parallel,          // if(parallel: x)
  if_target             // if(target: x)
};
} // unnamed namespace

struct OMPContext;
static OpenMPDirList SplitConstruct(OpenMPDirectiveKind);
static StringRef toString(OpenMPDirectiveKind k);
static StringRef toString(OpenMPClauseKind k);

static bool EmitSection(CodeGenFunction &, const OMPContext &);
static void EmitOMPStmtEnter(CodeGenFunction &, OMPContext &,
                             SmallVector<llvm::CallInst*, 4> &TokenList);
static void UpdateOMPStmtEnter(CodeGenFunction &, const OMPContext &,
                               SmallVector<llvm::CallInst*, 4> &TokenList,
                               const Stmt &body);
static void EmitOMPStmtExit(CodeGenFunction &, const OMPContext &,
                            SmallVector<llvm::CallInst*, 4> &TokenList);
static void EmitLoopExpressions(CodeGenFunction &, const OMPContext &, int);
static void EmitOMPClause(CodeGenFunction &, const OMPClause&,
                          OpenMPDirectiveKind, OMPContext &, int);
static bool markShared(CodeGenFunction &CGF, const OMPContext &, int,
                       BundleListTy &);
static bool privatizeAutos(CodeGenFunction&, const Stmt &, BundleListTy &);
static bool ApplyClause(OpenMPClauseKind, OpenMPDirectiveKind,
                        const OMPContext &, const OMPIfModifier);

struct OMPContext {
  OMPContext(const OMPExecutableDirective &s, const OpenMPDirList& splitDirs) :
    ompS(s), dirKind(ompS.getDirectiveKind()), SplitDirectives(splitDirs) {}

  // construct currently being processed
  const OMPExecutableDirective& ompS;

  // type of construct currently being processed
  const OpenMPDirectiveKind dirKind;

  // if we're processing a non-combined directive, then this is a list
  // containing just that one non-combined directive. Otherwise, this is a list
  // of (non-combined) directives that make up the combined directive.
  const OpenMPDirList SplitDirectives;

  // vector of operand bundle lists. Each list corresponds to one directive in
  // the SplitDirectives list and will be attached to an intrinsic CallInst.
  // Make this member mutable because it's the only one that gets often modified
  mutable SmallVector<BundleListTy, 4> BundleList;

  // accumulate all explicitly shared, private, firstprivate, lastprivate,
  // reduction, and linear variables in the construct
  SmallVector<VarDeclSet32, 4> ExplicitDSA;

  // save the iterator update if one is needed
  mutable SmallVector<Expr*, 4> iteratorUpdateExpr;
};


bool CodeGenFunction::EmitOMPStmt(const Stmt *S) {
  if (!isa<OMPExecutableDirective>(S)) return false;
  const OMPExecutableDirective &ompS = *cast<OMPExecutableDirective>(S);

  //
  // for now, generate annotations for 'parallel for' only
  if (ompS.getDirectiveKind() != clang::OMPD_parallel_for)  return false;

  // split any combined directives into a list of non-combined directives
  const OpenMPDirList SplitDirs = SplitConstruct(ompS.getDirectiveKind());

  // context object containing all necessary data structures that are used
  // during the processing of the current statement
  OMPContext C(ompS, SplitDirs);

  // begin OpenMP region
  OMPLexicalScope2 Scope(*this, ompS);

  // contains the region.entry tokens that may be updated later
  SmallVector<llvm::CallInst*, 4> TokenList;

  EmitOMPStmtEnter(*this, C, TokenList);

  // get the (body) statement associated to the construct
  const Stmt &body =
    isa<OMPLoopDirective>(&ompS)
      ? *dyn_cast<OMPLoopDirective>(&ompS)->getBody()
      : *cast<CapturedStmt>(ompS.getAssociatedStmt())->getCapturedStmt();


  // dump the body as is (for loops, just dump the loop body)
  CGM.getOpenMPRuntime().emitInlinedDirective(
      *this, C.dirKind, [&body](CodeGenFunction &CGF, PrePostActionTy &) {
        CGF.EmitStmt(&body);
      });


  UpdateOMPStmtEnter(*this, C, TokenList, body);

  EmitOMPStmtExit(*this, C, TokenList);

  return true;
}


static void EmitOMPStmtEnter(CodeGenFunction &CGF, OMPContext &C,
                             SmallVector<llvm::CallInst*, 4> &TokenList) {

  llvm::Function *regionEnter =
    llvm::Intrinsic::getDeclaration(&CGF.CGM.getModule(),
                                    llvm::Intrinsic::directive_region_entry);

  for (unsigned i=0; i<C.SplitDirectives.size(); i++) {
    C.BundleList.emplace_back();
    C.ExplicitDSA.emplace_back();
    const StringRef dirString = toString(C.SplitDirectives[i]);

    // set the directive kind
    C.BundleList[i].emplace_back(dirString, llvm::None);

    // if this is a loop directive, then create the bounds and step expressions
    if (isa<OMPLoopDirective>(C.ompS) &&
        isOpenMPLoopDirective(C.SplitDirectives[i])) {
      EmitLoopExpressions(CGF, C, i);
    }

    // insert a NOWAIT on the worksharing construct
    if (C.dirKind == OMPD_parallel_for) {
      if (C.SplitDirectives[i] == OMPD_for)
        C.BundleList[i].emplace_back("QUAL.OMP.NOWAIT", llvm::None);
    }

    // add any clauses
    for (const OMPClause* clause : C.ompS.clauses()) {
      EmitOMPClause(CGF, *clause, C.SplitDirectives[i], C, /*NLevel=*/i);
    }


    // generate:
    //   %0 = call token @llvm.directive.region.entry() [ "DIR.OMP.***"() ]
    //
    auto token = CGF.Builder.CreateCall(regionEnter, llvm::None, C.BundleList[i]);
    TokenList.push_back(token);
  }

  for(Expr* e: C.iteratorUpdateExpr){
    CGF.EmitLValue(e);
  }
}

static void UpdateOMPStmtEnter(CodeGenFunction &CGF, const OMPContext &C,
                               SmallVector<llvm::CallInst*, 4> &TokenList,
                               const Stmt &body) {
  // no llvm should be emitted at this point, so the IP is cleared
  const CGBuilderTy::InsertPoint IP = CGF.Builder.saveAndClearIP();

  const int MAX_SPLIT_DIRS = 4;
  const int IB = C.SplitDirectives.size() - 1;
  assert (IB <= MAX_SPLIT_DIRS);
  bool changed[MAX_SPLIT_DIRS] = {}; // default initialized to zero/false

  // emit non-static variables declared inside the construct as private;
  //   the inner most level is where the private bundle is attached.
  changed[IB] = privatizeAutos(CGF, body, C.BundleList[IB]);
  assert ((!changed[IB] || C.dirKind != clang::OMPD_target_data) && "TODO");

  // mark all referenced non-private variables inside the construct as shared
  for (int i = 0; i <= IB; i++)
    if (C.SplitDirectives[i] != clang::OMPD_target)
      changed[i] = markShared(CGF, C, /*NLevel=*/i, C.BundleList[i]) || changed[i];

  // update the directive.region.entry intrinsic with the extra information
  for (int i = 0; i <= IB; i++){
    if (changed[i]) {
      auto token2 = llvm::CallInst::Create(TokenList[i], C.BundleList[i]);
      llvm::ReplaceInstWithInst(TokenList[i], token2);
      TokenList[i] = token2;
    }
  }

  CGF.Builder.restoreIP(IP);
}

static void EmitOMPStmtExit(CodeGenFunction &CGF, const OMPContext &C,
                            SmallVector<llvm::CallInst*, 4> &TokenList) {

  // end OpenMP region
  SmallVector<BundleListTy, 4> ExitBundleList;
  llvm::Function *regionExit = llvm::Intrinsic::getDeclaration(&CGF.CGM.getModule(), llvm::Intrinsic::directive_region_exit);

  for (int i=C.SplitDirectives.size()-1; i>=0; i--) {
    BundleListTy OneExitBundle;
    const StringRef dirString = toString(C.SplitDirectives[i]);
    OneExitBundle.emplace_back(dirString, llvm::None);

    if (isa<OMPLoopDirective>(C.ompS) &&
        isOpenMPLoopDirective(C.SplitDirectives[i])) {
      unsigned CollapseVal = cast<OMPLoopDirective>(C.ompS).getCollapsedNumber();
      OneExitBundle.emplace_back(
        std::string("EXT.LOOP.END.") + llvm::itostr(CollapseVal), llvm::None);
    }
    ExitBundleList.push_back(OneExitBundle);

    // generate:
    //   call void @llvm.directive.region.exit(token %0) [ "DIR.OMP.***"() ]
    // where %0 is the region.entry call
    CGF.Builder.CreateCall(regionExit, {TokenList[i]}, ExitBundleList.back());
  }
}

static StringRef toString(OpenMPDirectiveKind k) {
  switch(k) {
  case clang::OMPD_parallel:                return "DIR.OMP.PARALLEL";
  case clang::OMPD_task:                    return "DIR.OMP.TASK";
  case clang::OMPD_single:                  return "DIR.OMP.SINGLE";
  case clang::OMPD_master:                  return "DIR.OMP.MASTER";
  case clang::OMPD_critical:                return "DIR.OMP.CRITICAL";
  case clang::OMPD_target:                  return "DIR.OMP.TARGET";
  case clang::OMPD_target_data:             return "DIR.OMP.TARGET.DATA";
  case clang::OMPD_teams:                   return "DIR.OMP.TEAMS";
  case clang::OMPD_for:                     return "DIR.OMP.LOOP";
  case clang::OMPD_sections:                return "DIR.OMP.SECTIONS";
  case clang::OMPD_section:                 return "DIR.OMP.SECTION";
  case clang::OMPD_distribute:              return "DIR.OMP.DISTRIBUTE";
  case clang::OMPD_distribute_parallel_for: return "DIR.OMP.DISTRIBUTE.PARLOOP";
  case clang::OMPD_simd:                    return "DIR.OMP.SIMD";
  case clang::OMPD_for_simd:                return "DIR.OMP.LOOP.SIMD";
  case clang::OMPD_distribute_simd:         return "DIR.OMP.DISTRIBUTE.SIMD";
  case clang::OMPD_distribute_parallel_for_simd:
                                            return "DIR.OMP.DISTRIBUTE.PARLOOP.SIMD";
  case clang::OMPD_taskloop:                return "DIR.OMP.TASKLOOP";
  case clang::OMPD_taskloop_simd:           return "DIR.OMP.TASKLOOP.SIMD";
  case clang::OMPD_target_update:           return "DIR.OMP.TARGET.UPDATE.DATA";
  case clang::OMPD_target_enter_data:       return "DIR.OMP.TARGET.ENTER.DATA";
  case clang::OMPD_target_exit_data:        return "DIR.OMP.TARGET.EXIT.DATA";
  case clang::OMPD_taskyield:               return "DIR.OMP.TASKYIELD";
  case clang::OMPD_barrier:                 return "DIR.OMP.BARRIER";
  case clang::OMPD_taskwait:                return "DIR.OMP.TASKWAIT";
  case clang::OMPD_flush:                   return "DIR.OMP.FLUSH";
  case clang::OMPD_threadprivate:           return "DIR.OMP.THREADPRIVATE";
  case clang::OMPD_ordered:                 return "DIR.OMP.ORDERED";
  default:
    assert(false && "add missing directive");
    return "INVALID";
  }
}

static StringRef toString(OpenMPClauseKind k) {
  switch(k) {
  case clang::OMPC_private:       return "QUAL.OMP.PRIVATE";
  case clang::OMPC_firstprivate:  return "QUAL.OMP.FIRSTPRIVATE";
  case clang::OMPC_lastprivate:   return "QUAL.OMP.LASTPRIVATE";
  case clang::OMPC_shared:        return "QUAL.OMP.SHARED";
  case clang::OMPC_collapse:      return "QUAL.OMP.COLLAPSE";
  case clang::OMPC_if:            return "QUAL.OMP.IF";
  case clang::OMPC_device:        return "QUAL.OMP.DEVICE";
  case clang::OMPC_num_threads:   return "QUAL.OMP.NUM_THREADS";
  case clang::OMPC_num_teams:     return "QUAL.OMP.NUM_TEAMS";
  case clang::OMPC_safelen:       return "QUAL.OMP.SAFELEN";
  case clang::OMPC_simdlen:       return "QUAL.OMP.SIMDLEN";
  case clang::OMPC_thread_limit:  return "QUAL.OMP.THREAD_LIMIT";
  case clang::OMPC_untied:        return "QUAL.OMP.UNTIED";
  case clang::OMPC_mergeable:     return "QUAL.OMP.MERGEABLE";
  case clang::OMPC_final:         return "QUAL.OMP.FINAL";
  case clang::OMPC_nowait:        return "QUAL.OMP.NOWAIT";
  case clang::OMPC_proc_bind:     return "QUAL.OMP.PROC_BIND";
  case clang::OMPC_copyprivate:   return "QUAL.OMP.COPYPRIVATE";
  case clang::OMPC_dist_schedule: return "QUAL.OMP.DIST_SCHEDULE.STATIC";
  case clang::OMPC_ordered:       return "QUAL.OMP.ORDERED";
  case clang::OMPC_flush:         return "QUAL.OMP.FLUSH";
  case clang::OMPC_copyin:        return "QUAL.OMP.COPYIN";
  default:
    assert(0 && "unimplemented");
    return "UNKNOWN";
  }
}

static StringRef toString(OpenMPClauseKind k, OpenMPLinearClauseKind lck) {
  assert(k == OMPC_linear && "expected linear clause");
  switch (lck) {
  case OMPC_LINEAR_val:  return "QUAL.OMP.LINEAR.VAL";
  case OMPC_LINEAR_uval: return "QUAL.OMP.LINEAR.UVAL";
  case OMPC_LINEAR_ref:  return "QUAL.OMP.LINEAR.REF";
  default:
    assert(0 && "unknown linear clause kind");
    return "UNKNOWN";
  }
}

static StringRef extractOperator(const OMPReductionClause &c, bool isSigned) {
  DeclarationName op = c.getNameInfo().getName();
  StringRef str = StringRef(op.getAsString());

  if(str == "min"){
    return isSigned ? "QUAL.OMP.REDUCTION.MIN.S" : "QUAL.OMP.REDUCTION.MIN";
  } else if(str == "max"){
    return isSigned ? "QUAL.OMP.REDUCTION.MAX.S" : "QUAL.OMP.REDUCTION.MAX";
  }

  assert(str.startswith("operator") && "expected operator to extract");
  str = str.substr(8);

  if(str == "+"){
    return isSigned ? "QUAL.OMP.REDUCTION.ADD.S" : "QUAL.OMP.REDUCTION.ADD";
  } else if(str == "-"){
    return isSigned ? "QUAL.OMP.REDUCTION.SUB.S" : "QUAL.OMP.REDUCTION.SUB";
  } else if(str == "*"){
    return isSigned ? "QUAL.OMP.REDUCTION.MUL.S" : "QUAL.OMP.REDUCTION.MUL";
  } else if(str == "||"){
    return isSigned ? "QUAL.OMP.REDUCTION.OR.S" : "QUAL.OMP.REDUCTION.OR";
  } else if(str == "&&"){
    return isSigned ? "QUAL.OMP.REDUCTION.AND.S" : "QUAL.OMP.REDUCTION.AND";
  } else if(str == "|"){
    return isSigned ? "QUAL.OMP.REDUCTION.BOR.S" : "QUAL.OMP.REDUCTION.BOR";
  } else if(str == "&"){
    return isSigned ? "QUAL.OMP.REDUCTION.BAND.S" : "QUAL.OMP.REDUCTION.BAND";
  } else if(str == "^"){
    return isSigned ? "QUAL.OMP.REDUCTION.BXOR.S" : "QUAL.OMP.REDUCTION.BXOR";
  } else {
    // doesn't handle user defined operators
    assert(false && "unexpected operator");
  }
}

static void handleDefaultClause(CodeGenFunction &CGF, const OMPDefaultClause &c, BundleListTy &bundle){
  OpenMPDefaultClauseKind kind = c.getDefaultKind();
  switch(kind){
  case clang::OMPC_DEFAULT_shared:
    bundle.emplace_back("QUAL.OMP.DEFAULT.SHARED", ArrayRef<llvm::Value*>());
    break;
  case clang::OMPC_DEFAULT_none:
    bundle.emplace_back("QUAL.OMP.DEFAULT.NONE", ArrayRef<llvm::Value*>());
    break;
  default:
    assert(0 && "unexpected parameter for default clause");
  }
}

static void handleScheduleClause(CodeGenFunction &CGF, const OMPScheduleClause &c, BundleListTy &bundle){
  OpenMPScheduleClauseKind kind = c.getScheduleKind();
  ArrayRef<llvm::Value*> ar = c.getChunkSize() ?
      ArrayRef<llvm::Value*>(CGF.EmitScalarExpr(c.getChunkSize())) :
      ArrayRef<llvm::Value*>();

  switch(kind){
  case clang::OMPC_SCHEDULE_static:
    bundle.emplace_back("QUAL.OMP.SCHEDULE.STATIC", ar);
    break;
  case clang::OMPC_SCHEDULE_dynamic:
    bundle.emplace_back("QUAL.OMP.SCHEDULE.DYNAMIC", ar);
    break;
  case clang::OMPC_SCHEDULE_guided:
    bundle.emplace_back("QUAL.OMP.SCHEDULE.GUIDED", ar);
    break;
  case clang::OMPC_SCHEDULE_auto:
    bundle.emplace_back("QUAL.OMP.SCHEDULE.AUTO", ar);
    break;
  case clang::OMPC_SCHEDULE_runtime:
    bundle.emplace_back("QUAL.OMP.SCHEDULE.RUNTIME", ar);
    break;
  default:
    assert(0 && "unexpected parameter for schedule clause");
  }
}

template <class CL>
static void handleClauseWithVars(CodeGenFunction &CGF, const OMPVarListClause<CL> &c, BundleListTy &bundle) {
  SmallVector<llvm::Value*, 8> varList;
  for (auto I = c.varlist_begin(), E = c.varlist_end();  I != E;  ++I)
    {
    const Expr* expr = *I;
    llvm::Value* v = CGF.EmitLValue(expr).getPointer();
    varList.push_back(v);
    }

  bundle.emplace_back(toString(c.getClauseKind()), varList);
}

/// \brief In addition to performing what handleClauseWithVars(,,) does,
/// collect the clause's list item declarations in
///  Context::mapped if the clause is on a target region, or
///  Context::ExplicitDSA otherwise
///
template <class CL>
static void handleClauseAndAccum(CodeGenFunction &CGF,
                                 const OMPVarListClause<CL> &c, OMPContext &C,
                                 int NLevel) {
  SmallVector<llvm::Value*, 8> varList;
  clang::OpenMPClauseKind clause = c.getClauseKind();

  if (clause == clang::OMPC_reduction) {
    //
    // we want to emit the signed integer type list items in a separate bundle
    // from the rest of the list items.
    SmallVector<llvm::Value*, 8> signedVarList;
    for (auto I = c.varlist_begin(), E = c.varlist_end();  I != E;  ++I) {
      const Expr* expr = *I;
      if (isa<DeclRefExpr>(expr)) {
        const VarDecl *VD =
          dyn_cast_or_null<VarDecl>(cast<DeclRefExpr>(expr)->getDecl());
        C.ExplicitDSA[NLevel].insert(VD);
        llvm::Value* v = CGF.EmitLValue(expr).getPointer();

        if (VD->getType().getTypePtr()->isSignedIntegerType())
          signedVarList.push_back(v);
        else
          varList.push_back(v);
      }
      else
        assert(false && "unimplemented");
    }
    if (!varList.empty()) {
      StringRef tag = extractOperator(cast<OMPReductionClause>(c),
                                      false /*isSigned*/);
      C.BundleList[NLevel].emplace_back(tag, varList);
    }
    if (!signedVarList.empty()) {
      StringRef tag = extractOperator(cast<OMPReductionClause>(c),
                                      true/*isSigned*/);
      C.BundleList[NLevel].emplace_back(tag, signedVarList);
    }
    return;
  }
  //
  // else do the default work
  assert((clause==clang::OMPC_firstprivate || clause==clang::OMPC_private ||
          clause==clang::OMPC_lastprivate  || clause==clang::OMPC_linear ||
          clause==clang::OMPC_shared) &&
          "this should be called for DSA clauses only");
  for (auto I = c.varlist_begin(), E = c.varlist_end();  I != E;  ++I) {
    const Expr* expr = *I;
    if (isa<DeclRefExpr>(expr))
      C.ExplicitDSA[NLevel].insert(dyn_cast_or_null<VarDecl>(
                                     cast<DeclRefExpr>(expr)->getDecl()
                                   ));
    else
      assert(false && "unimplemented");

    llvm::Value* v = CGF.EmitLValue(expr).getPointer();
    varList.push_back(v);
  }

  StringRef tag;
  if (clause == clang::OMPC_linear) {
    auto *StepExpr = cast<OMPLinearClause>(c).getStep();
    auto lstep = (StepExpr == nullptr)
                     ? llvm::ConstantInt::get(CGF.CGM.Int64Ty, 1)
                     : CGF.EmitScalarExpr(StepExpr);
    varList.push_back(lstep);

    tag = toString(clause, cast<OMPLinearClause>(c).getModifier());
  } else
    tag = toString(clause);

  C.BundleList[NLevel].emplace_back(tag, varList);
}

static void handleClauseWithExpr(CodeGenFunction &CGF, const OMPClause &clause, BundleListTy &bundle) {
  const Expr *expr = nullptr;

  switch(clause.getClauseKind()) {
  case clang::OMPC_collapse:
    expr = cast<OMPCollapseClause>(clause).getNumForLoops();
    break;
  case clang::OMPC_if:
    expr = cast<OMPIfClause>(clause).getCondition();
    break;
  case clang::OMPC_num_threads:
    expr = cast<OMPNumThreadsClause>(clause).getNumThreads();
    break;
  default:
    assert(0 && "unimplemented");
  }

  if (expr) {
    llvm::Value *v = CGF.EmitScalarExpr(expr);
    bundle.emplace_back(toString(clause.getClauseKind()),
                        ArrayRef<llvm::Value*>(v));
  } else {
    bundle.emplace_back(toString(clause.getClauseKind()), llvm::None);
  }
}

static OMPIfModifier handleIfClause(const OMPIfClause &IC) {
  auto IfNameModifier = IC.getNameModifier();
  auto IM = OMPIfModifier::no_if;

  switch (IfNameModifier) {
  case clang::OMPD_parallel:
    IM = OMPIfModifier::if_parallel;
    break;
  case clang::OMPD_target:
    IM = OMPIfModifier::if_target;
    break;
  default:
    IM = OMPIfModifier::if_none;
  }
  return IM;
}

static void EmitOMPClause(CodeGenFunction &CGF, const OMPClause &clause,
                          OpenMPDirectiveKind ODK, OMPContext &C, int NLevel) {
  OpenMPClauseKind Kind = clause.getClauseKind();
  auto IM = OMPIfModifier::no_if;

  if (Kind == OMPC_if)
    IM = handleIfClause(cast<OMPIfClause>(clause));

  //
  // early exit if we're processing a sub-construct of a combined construct
  // and the clause does not apply
  if (!ApplyClause(Kind, ODK, C, IM))
    return;

  switch (Kind) {
  //
  // DSA clauses
  case clang::OMPC_private:
    handleClauseAndAccum(CGF, cast<OMPPrivateClause>(clause), C, NLevel);
    break;

  case clang::OMPC_firstprivate:
    handleClauseAndAccum(CGF, cast<OMPFirstprivateClause>(clause), C, NLevel);
    break;

  case clang::OMPC_lastprivate:
    handleClauseAndAccum(CGF, cast<OMPLastprivateClause>(clause), C, NLevel);
    break;

  case clang::OMPC_shared:
    handleClauseAndAccum(CGF, cast<OMPSharedClause>(clause), C, NLevel);
    break;

  case clang::OMPC_reduction:
    handleClauseAndAccum(CGF, cast<OMPReductionClause>(clause), C, NLevel);
    break;

  case clang::OMPC_linear:
    handleClauseAndAccum(CGF, cast<OMPLinearClause>(clause), C, NLevel);
    break;

  //
  // non-DSA clauses

  case clang::OMPC_default:
    handleDefaultClause(CGF, cast<OMPDefaultClause>(clause), C.BundleList[NLevel]);
    break;

  case clang::OMPC_schedule:
    handleScheduleClause(CGF, cast<OMPScheduleClause>(clause), C.BundleList[NLevel]);
    break;

  // clauses with scalar expressions
  case clang::OMPC_if:
  case clang::OMPC_collapse:
  case clang::OMPC_num_threads:
    handleClauseWithExpr(CGF, clause, C.BundleList[NLevel]);
    break;

  // proc_bind clauses
  case clang::OMPC_proc_bind:
    C.BundleList[NLevel].emplace_back(
      toString(clause.getClauseKind()).str() + std::string(".") +
          StringRef(getOpenMPSimpleClauseTypeName(
                        OMPC_proc_bind,
                        cast<OMPProcBindClause>(clause).getProcBindKind()))
              .upper(),
      llvm::None);
    break;

  default:
    llvm::errs() << "XYZ: unimplemented clause " << clause.getClauseKind() << "\n";
    assert(0 && "unimplemented clause");
    break;
  }
}

static void EmitLoopExpressions(CodeGenFunction &CGF, const OMPContext &C,
                                int NLevel) {
  const OMPLoopDirective &ompS = cast<OMPLoopDirective>(C.ompS);
  const Stmt *Body = ompS.getAssociatedStmt()->IgnoreContainers(true);
  //
  // now we expect C number of nested for-statements, where C is the constant
  // on the collapse clause
  const unsigned Collapse = ompS.getCollapsedNumber();
  for (unsigned Cnt = 0; Cnt < Collapse; ++Cnt) {
    Body = Body->IgnoreContainers();
    assert (isa<ForStmt>(Body) &&
          "loop directives must have an associated loop(s)");

    // process the declaration in the loop header
    const ForStmt* forStmt = dyn_cast<ForStmt>(Body);
    if (isa<DeclStmt>(forStmt->getInit())){
      const DeclStmt* iteratorDeclStmt = dyn_cast<DeclStmt>(forStmt->getInit());
      assert (iteratorDeclStmt->isSingleDecl());
      const VarDecl* iteratorVarDecl =
        dyn_cast<VarDecl>(iteratorDeclStmt->getSingleDecl())->getDefinition();
      CGF.EmitAutoVarAlloca(*iteratorVarDecl);
    }

    llvm::Value *init = nullptr, *iv = nullptr;
    llvm::Value *final = nullptr, *inc = nullptr;

    // 1. iv
    const Expr* ivExpr = ompS.AnnoIVRefs()[Cnt];
    assert (isa<DeclRefExpr>(ivExpr));

    const DeclRefExpr* ivDeclRefExpr = dyn_cast<DeclRefExpr>(ivExpr);
    const ValueDecl* ivVarDecl = ivDeclRefExpr->getDecl();
    //if this loop uses an iterator
    if (!CGF.inLocalDeclMap(dyn_cast<VarDecl>(ivVarDecl))){
      CGF.EmitAutoVarDecl(*dyn_cast<VarDecl>(ivVarDecl));
    }

    iv = CGF.EmitLValue(ivExpr).getPointer();

    // 3. initial value
    const Expr* initExpr = ompS.AnnoInits()[Cnt];
    init = CGF.EmitScalarExpr(initExpr);

    // 4. final value
    const Expr* finalExpr = ompS.AnnoFinals()[Cnt];
    final = CGF.EmitScalarExpr(finalExpr);

    // 5. step expr
    const Expr* stepExpr = ompS.AnnoSteps()[Cnt];
    inc = CGF.EmitScalarExpr(stepExpr);

    // 6. create the bundle:
    //   "EXT.LOOP.*"(iv, init, final, inc)
    SmallVector<llvm::Value*, 4> varList;
    varList.push_back(iv);
    varList.push_back(init);
    varList.push_back(final);
    varList.push_back(inc);
    C.BundleList[NLevel].emplace_back("EXT.LOOP", varList);

    const Expr* startExpr = ompS.AnnoIterStart()[Cnt];
    if(startExpr){
      const CallExpr* startCallExpr = cast<CallExpr>(startExpr);
      const DeclRefExpr* startDeclRefExpr = cast<DeclRefExpr>(startCallExpr->getArg(0));
      const ValueDecl* startValueDecl = startDeclRefExpr->getDecl();
      const Decl* startDecl = cast<Decl>(startValueDecl);
      CGF.EmitDecl(*startDecl);
//      llvm::Value* start =
      CGF.EmitLValue(startExpr).getPointer();
      C.iteratorUpdateExpr.push_back(ompS.AnnoIterUpdate()[Cnt]);
    }

    //
    // 7. privatize iteration variable
    C.BundleList[NLevel].emplace_back(toString(clang::OMPC_private),
                                      ArrayRef<llvm::Value*>(iv));

    // prepare for the next iteration, if any
    Body = cast<ForStmt>(Body)->getBody();
  }
}

static bool markShared(CodeGenFunction &CGF, const OMPContext &C,
                       int NLevel, BundleListTy &bundle) {

  struct markSharedVisitor : public RecursiveASTVisitor<markSharedVisitor> {
    markSharedVisitor(const VarDeclSet32 &explict) : Explicit(explict) {}
    bool shouldVisitImplicitCode() const {
      return true;
    }
    bool TraverseStmt(Stmt *S) {
      if (!S)
        return true;
      if (isa<OMPExecutableDirective>(S)) {
        for (auto *C : cast<OMPExecutableDirective>(S)->clauses()) {
          TraverseOMPClause(C);
        }
        return true;
      }
      return RecursiveASTVisitor<markSharedVisitor>::TraverseStmt(S);
    }
    bool VisitDeclRefExpr(DeclRefExpr *dre) {
      if (const VarDecl *VD = dyn_cast_or_null<VarDecl>(dre->getDecl())) {
        //
        // only collect globals and local statics; non-globals must have been
        // captured already.
        // NOTE: if we want to start collecting non-globals, then we need
        // to make sure they're not automatics declared inside the current
        // region, in which case they ought to be private!!
        // Static-locals declared inside the current region will not be captured
        // but we need to make them shared with respect to the current constrct.
        if (VD->hasLinkage() || VD->isStaticDataMember() || VD->isStaticLocal())
          if (Explicit.count(VD) == 0)
            ImplicitShared.insert(VD);
      }
      return true;
    }
    const VarDeclSet32 &Explicit; // vars appearing explicitly on DSA clauses
    VarDeclSet32 ImplicitShared; // accumulate all implicitly shared vars here
  };

  // traverse AST of the directive and collect implicitly shared vars
  markSharedVisitor V(C.ExplicitDSA[NLevel]);
  const Stmt &body =
      isa<OMPLoopDirective>(&C.ompS)
        ? *dyn_cast<OMPLoopDirective>(&C.ompS)->getBody()
        : *cast<CapturedStmt>(C.ompS.getAssociatedStmt())->getCapturedStmt();
  V.TraverseStmt(const_cast<Stmt*>(&body));

  //
  // walk the captured variables and add them to the shared list
  const CapturedStmt &S = *cast<CapturedStmt>(C.ompS.getAssociatedStmt());
  for (const CapturedStmt::Capture &Cap : S.captures()) {
    if (Cap.capturesVariable() || Cap.capturesVariableByCopy()) {
      VarDecl *VD = Cap.getCapturedVar();

      // add to the shared list if it's not explicitly marked as shared
      if (C.ExplicitDSA[NLevel].count(VD) == 0)
        V.ImplicitShared.insert(VD);
    }
  }

  // lower the VarDecls to llvm values
  SmallVector<llvm::Value*, 32> valueList;
  for (auto it = V.ImplicitShared.begin(), end = V.ImplicitShared.end();
       it != end;
       ++it) {
    const VarDecl* VD = *it;
    //
    // it's either a global static
    if (VD->hasLinkage() || VD->isStaticDataMember())
      valueList.push_back(CGF.CGM.GetAddrOfGlobalVar(VD));
    //
    // or a local static
    // Note: some declarations are inside a dead-code path and so they are not
    // CG'ed. E.g.
    //  if (0) {
    //    static int x = 0;
    //  }
    else if (CGF.inLocalDeclMap(VD))
      valueList.push_back(CGF.GetAddrOfLocalVar(VD).getPointer());
  }

  if (!valueList.empty()) {
    C.BundleList[NLevel].emplace_back(toString(clang::OMPC_shared), valueList);
    return true;
  }
  return false;
}


// \brief Append all automatics to a private clause bundle
static bool privatizeAutos(CodeGenFunction &CGF, const Stmt &body,
                           BundleListTy &bundle) {
  //
  // \brief visit the body of the construct but not any nested constructs
  struct MarkPrivVisitor : public RecursiveASTVisitor<MarkPrivVisitor> {
    MarkPrivVisitor(CodeGenFunction &_CGF) : CGF(_CGF) {}
    SmallVector<llvm::Value*, 32> ImplicitPriv;
    CodeGenFunction &CGF;

    bool shouldVisitImplicitCode() const { return true; }
    bool TraverseStmt(Stmt *S) {
      if (!S || isa<OMPExecutableDirective>(S))
        return true;
      return RecursiveASTVisitor<MarkPrivVisitor>::TraverseStmt(S);
    }
    bool VisitDeclStmt(DeclStmt *DS) {
      for (auto *I : DS->decls()) {
        VarDecl *VD = cast<VarDecl>(I);
        //
        // static-locals are not implicitly private
        if (VD->isStaticLocal())
          break;

        //
        // do not privatize automatic VLA's because:
        //  1) the alloca is inside the OpenMP region, hence cannot refer to it
        //  2) L2W translates VL alloca to __builtin_alloca and stores the
        //     result in a private pointer-type variable
        if (VD->getType()->isVariablyModifiedType())
          break;

        // some declarations are inside a dead-code path, so you won't find
        // corresponding llvm alloca's for them. E.g.
        //  if (0) {
        //    int x = 0;
        //  }
        if (CGF.inLocalDeclMap(VD)) {
          Address LVarAddr = CGF.GetAddrOfLocalVar(VD);
          ImplicitPriv.push_back(LVarAddr.getPointer());
        }
      }
      return true;
    }
  } V(CGF);

  V.TraverseStmt(const_cast<Stmt*>(&body));

  if (!V.ImplicitPriv.empty()) {
    bundle.emplace_back(toString(clang::OMPC_private), V.ImplicitPriv);
    return true;
  }
  return false;
}

static constexpr bool SupportedCombinedDirectives(OpenMPDirectiveKind k) {
  switch (k) {
  case clang::OMPD_parallel_sections:
  case clang::OMPD_parallel_for:
  case clang::OMPD_parallel_for_simd:
  case clang::OMPD_target_parallel:
  case clang::OMPD_target_teams:
  case clang::OMPD_target_simd:
  case clang::OMPD_target_parallel_for:
  case clang::OMPD_target_parallel_for_simd:
  case clang::OMPD_target_teams_distribute:
  case clang::OMPD_target_teams_distribute_simd:
  case clang::OMPD_target_teams_distribute_parallel_for:
  case clang::OMPD_target_teams_distribute_parallel_for_simd:
  case clang::OMPD_teams_distribute:
  case clang::OMPD_teams_distribute_simd:
  case clang::OMPD_teams_distribute_parallel_for:
  case clang::OMPD_teams_distribute_parallel_for_simd:
    return true;
  default:
    return false;
  }
}

/// Return true if the given clause \a clause is applicable to directive \a SD
/// given that it might be part of the original combined directive OMPC.dirKind
/// If SD == OMPC.dirKind then we are processing a non-combined directive
static bool ApplyClause(OpenMPClauseKind clause, OpenMPDirectiveKind SD,
                        const OMPContext &OMPC, OMPIfModifier IM) {

  // 1. for non-combined constructs, the clause applies by default (otherwise
  //    it's a syntax error and Sema/Parser would have caught it)
  if (OMPC.dirKind == SD)
    return true;

  const OpenMPDirectiveKind combined = OMPC.dirKind;
  assert (SupportedCombinedDirectives(combined));

  // 2. deal with the special case of the 'if' clause
  if (clause == OMPC_if) {
    switch (IM) {
    case OMPIfModifier::if_parallel:
      if (SD == OMPD_parallel)
        return true;
      break;
    case OMPIfModifier::if_target:
      if (SD == OMPD_target)
        return true;
      break;
    // TODO deal with other directive names
    case OMPIfModifier::if_none:
      if (SD == OMPD_target || SD == OMPD_parallel)
        return true;
      break;
    default:
      assert(false && "unexpected name directive modifier");
      break;
    }
    return false;
  }

  // 3. this is part of a combined directive
  switch (SD) {
    default:
      llvm::errs() << "unexpected directive " << getOpenMPDirectiveName(SD);
      assert(false && "unexpected directive");

    case OMPD_parallel:
      switch (clause) {
        default: break;
        case OMPC_reduction:
        case OMPC_private:
        case OMPC_firstprivate:
        case OMPC_copyin:
        case OMPC_shared:
        case OMPC_default:
        case OMPC_num_threads:
        case OMPC_proc_bind:
          return true;
      }
      break;

    case OMPD_simd:
      switch (clause) {
        default: break;
        case OMPC_reduction:
        case OMPC_private:
        case OMPC_lastprivate:
        case OMPC_collapse:
        case OMPC_safelen:
        case OMPC_simdlen:
        case OMPC_aligned:
        case OMPC_linear:
          return true;
      }
      break;

    case OMPD_for:
      switch (clause) {
        default: break;
        case OMPC_reduction:
        case OMPC_private:
        case OMPC_firstprivate:
        case OMPC_lastprivate:
        case OMPC_linear:
        case OMPC_schedule:
        case OMPC_collapse:
        case OMPC_nowait:
        case OMPC_ordered:
          return true;
      }
      break;

    case OMPD_sections:
      switch (clause) {
        default: break;
        case OMPC_reduction:
        case OMPC_private:
        case OMPC_firstprivate:
        case OMPC_lastprivate:
        case OMPC_nowait:
          return true;
      }
      break;

    case OMPD_target:
      switch (clause) {
        default: break;
        case OMPC_private:
        case OMPC_firstprivate:
        case OMPC_nowait:
        case OMPC_map:
        case OMPC_device:
        case OMPC_depend:
        case OMPC_is_device_ptr:
          return true;
      }
      break;

    case OMPD_teams:
      switch (clause) {
        default: break;
        case OMPC_reduction:
        case OMPC_private:
        case OMPC_firstprivate:
        case OMPC_default:
        case OMPC_num_teams:
        case OMPC_thread_limit:
        case OMPC_shared:
          return true;
      }
      break;

    case OMPD_for_simd:
      switch (clause) {
        default: break;
        case OMPC_reduction:
        case OMPC_private:
        case OMPC_firstprivate:
        case OMPC_lastprivate:
        case OMPC_linear:
        case OMPC_collapse:
        case OMPC_schedule:
        case OMPC_ordered:
        case OMPC_safelen:
        case OMPC_simdlen:
        case OMPC_aligned:
        case OMPC_nowait:
          return true;
      }
      break;

    case OMPD_distribute:
      switch (clause) {
        default: break;
        case OMPC_private:
        case OMPC_firstprivate:
        case OMPC_lastprivate:
        case OMPC_collapse:
        case OMPC_dist_schedule:
          return true;
      }
      break;

    case OMPD_distribute_parallel_for:
      switch (clause) {
        default: break;
        case OMPC_reduction:
        case OMPC_private:
        case OMPC_firstprivate:
        case OMPC_lastprivate:
        case OMPC_default:
        case OMPC_shared:
        case OMPC_schedule:
        case OMPC_collapse:
        case OMPC_dist_schedule:
        case OMPC_proc_bind:
        case OMPC_copyin:
        case OMPC_num_threads:
          return true;
      }
      break;

    case OMPD_distribute_parallel_for_simd:
      switch (clause) {
        default: break;
        case OMPC_reduction:
        case OMPC_private:
        case OMPC_firstprivate:
        case OMPC_lastprivate:
        case OMPC_schedule:
        case OMPC_collapse:
        case OMPC_dist_schedule:
        case OMPC_proc_bind:
        case OMPC_copyin:
        case OMPC_default:
        case OMPC_num_threads:
        case OMPC_shared:
        case OMPC_safelen:
        case OMPC_simdlen:
        case OMPC_aligned:
        case OMPC_linear:
          return true;
      }
      break;

    case OMPD_distribute_simd:
      switch (clause) {
        default: break;
        case OMPC_reduction:
        case OMPC_private:
        case OMPC_firstprivate:
        case OMPC_lastprivate:
        case OMPC_collapse:
        case OMPC_dist_schedule:
        case OMPC_default:
        case OMPC_shared:
        case OMPC_safelen:
        case OMPC_simdlen:
        case OMPC_aligned:
        case OMPC_linear:
          return true;
      }
      break;
  }
  return false;
}


static OpenMPDirList SplitConstruct(OpenMPDirectiveKind dirKind) {

  OpenMPDirList explodedDirectives;

  switch (dirKind) {
  // block constructs
  case clang::OMPD_parallel:
  case clang::OMPD_task:
  case clang::OMPD_single:
  case clang::OMPD_master:
  case clang::OMPD_critical:
  case clang::OMPD_target:
  case clang::OMPD_target_data:
  case clang::OMPD_teams:
  case clang::OMPD_sections:
  case clang::OMPD_section:
  case clang::OMPD_ordered:
    explodedDirectives.push_back(dirKind);
    break;

  // loop constructs
  case clang::OMPD_for:
  case clang::OMPD_distribute:
  case clang::OMPD_distribute_parallel_for:
  case clang::OMPD_simd:
  case clang::OMPD_for_simd:
  case clang::OMPD_distribute_simd:
  case clang::OMPD_distribute_parallel_for_simd:
  case clang::OMPD_taskloop:
  case clang::OMPD_taskloop_simd:
    explodedDirectives.push_back(dirKind);
    break;

  // standalone with clauses
  case clang::OMPD_target_update:
  case clang::OMPD_target_enter_data:
  case clang::OMPD_target_exit_data:
  // standalone
  case clang::OMPD_taskyield:
  case clang::OMPD_barrier:
  case clang::OMPD_taskwait:
  case clang::OMPD_flush:
    explodedDirectives.push_back(dirKind);
    break;

  // combined constructs
  case clang::OMPD_parallel_for:
    explodedDirectives.push_back(OMPD_parallel);
    explodedDirectives.push_back(OMPD_for);
    break;
  case clang::OMPD_target_parallel:
    explodedDirectives.push_back(OMPD_target);
    explodedDirectives.push_back(OMPD_parallel);
    break;
  case clang::OMPD_target_teams:
    explodedDirectives.push_back(OMPD_target);
    explodedDirectives.push_back(OMPD_teams);
    break;
  case clang::OMPD_teams_distribute:
    explodedDirectives.push_back(OMPD_teams);
    explodedDirectives.push_back(OMPD_distribute);
    break;
  case clang::OMPD_parallel_for_simd:
    explodedDirectives.push_back(OMPD_parallel);
    explodedDirectives.push_back(OMPD_for_simd);
    break;
  case clang::OMPD_target_simd:
    explodedDirectives.push_back(OMPD_target);
    explodedDirectives.push_back(OMPD_simd);
    break;
  case clang::OMPD_parallel_sections:
    explodedDirectives.push_back(OMPD_parallel);
    explodedDirectives.push_back(OMPD_sections);
    break;
  case clang::OMPD_teams_distribute_simd:
    explodedDirectives.push_back(OMPD_teams);
    explodedDirectives.push_back(OMPD_distribute_simd);
    break;
  case clang::OMPD_target_parallel_for:
    explodedDirectives.push_back(OMPD_target);
    explodedDirectives.push_back(OMPD_parallel);
    explodedDirectives.push_back(OMPD_for);
    break;
  case clang::OMPD_target_parallel_for_simd:
    explodedDirectives.push_back(OMPD_target);
    explodedDirectives.push_back(OMPD_parallel);
    explodedDirectives.push_back(OMPD_for_simd);
    break;
  case clang::OMPD_target_teams_distribute:
    explodedDirectives.push_back(OMPD_target);
    explodedDirectives.push_back(OMPD_teams);
    explodedDirectives.push_back(OMPD_distribute);
    break;
  case clang::OMPD_target_teams_distribute_simd:
    explodedDirectives.push_back(OMPD_target);
    explodedDirectives.push_back(OMPD_teams);
    explodedDirectives.push_back(OMPD_distribute_simd);
    break;
  case clang::OMPD_teams_distribute_parallel_for:
    explodedDirectives.push_back(OMPD_teams);
    explodedDirectives.push_back(OMPD_distribute_parallel_for);
    break;
  case clang::OMPD_target_teams_distribute_parallel_for:
    explodedDirectives.push_back(OMPD_target);
    explodedDirectives.push_back(OMPD_teams);
    explodedDirectives.push_back(OMPD_distribute_parallel_for);
    break;
  case clang::OMPD_teams_distribute_parallel_for_simd:
    explodedDirectives.push_back(OMPD_teams);
    explodedDirectives.push_back(OMPD_distribute_parallel_for_simd);
    break;
  case clang::OMPD_target_teams_distribute_parallel_for_simd:
    explodedDirectives.push_back(OMPD_target);
    explodedDirectives.push_back(OMPD_teams);
    explodedDirectives.push_back(OMPD_distribute_parallel_for_simd);
    break;

  // misc
  case clang::OMPD_threadprivate:
  case clang::OMPD_atomic:
  case clang::OMPD_cancel:
  default:
    llvm::errs() << "XYZ: unimplemented directive" << dirKind;
    assert(false && !"unimplemented directive");
  }

  return explodedDirectives;
}
