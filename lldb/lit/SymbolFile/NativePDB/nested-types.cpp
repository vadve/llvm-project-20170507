// clang-format off
// REQUIRES: lld

// Test various interesting cases for AST reconstruction.
// RUN: %build --compiler=clang-cl --nodefaultlib -o %t.exe -- %s 
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb -f %t.exe -s \
// RUN:     %p/Inputs/nested-types.lldbinit 2>&1 | FileCheck %s

struct S {
  struct NestedStruct {
    int A = 0;
    int B = 1;
  };
  int C = 2;
  int D = 3;
  using VoidPtrT = void *;
  VoidPtrT DD = nullptr;
};
struct T {
  using NestedTypedef = int;
  using NestedTypedef2 = S;

  struct NestedStruct {
    int E = 4;
    int F = 5;
  };

  using NestedStructAlias = NestedStruct;
  using NST = S::NestedStruct;

  NestedTypedef NT = 4;

  using U = struct {
    int G = 6;
    int H = 7;
  };
};

template<typename Param>
class U {
public:
  // See llvm.org/pr39607.  clang-cl currently doesn't emit an important debug
  // info record for nested template instantiations, so we can't reconstruct
  // a proper DeclContext hierarchy for these.  As such, U<X>::V<Y> will show up
  // in the global namespace.
  template<typename Param>
  struct V {
    Param I = 8;
    Param J = 9;

    using W = T::NestedTypedef;
    using X = U<int>;
  };

  struct W {
    Param M = 12;
    Param N = 13;
  };
  Param K = 10;
  Param L = 11;
  using Y = V<int>;
  using Z = V<T>;
};

constexpr S GlobalA;
constexpr S::NestedStruct GlobalB;
constexpr T GlobalC;
constexpr T::NestedStruct GlobalD;
constexpr T::U GlobalE;
constexpr U<int> GlobalF;
constexpr U<int>::V<int> GlobalG;
constexpr U<int>::W GlobalH;


int main(int argc, char **argv) {
  return 0;
}



// CHECK: (lldb) target variable -T GlobalA
// CHECK: (const S) GlobalA = {
// CHECK:   (int) C = 2
// CHECK:   (int) D = 3
// CHECK:   (void *) DD = 0x00000000
// CHECK: }
// CHECK: (lldb) target variable -T GlobalB
// CHECK: (const S::NestedStruct) GlobalB = {
// CHECK:   (int) A = 0
// CHECK:   (int) B = 1
// CHECK: }
// CHECK: (lldb) target variable -T GlobalC
// CHECK: (const T) GlobalC = {
// CHECK:   (int) NT = 4
// CHECK: }
// CHECK: (lldb) target variable -T GlobalD
// CHECK: (const T::NestedStruct) GlobalD = {
// CHECK:   (int) E = 4
// CHECK:   (int) F = 5
// CHECK: }
// CHECK: (lldb) target variable -T GlobalE
// CHECK: (const T::U) GlobalE = {
// CHECK:   (int) G = 6
// CHECK:   (int) H = 7
// CHECK: }
// CHECK: (lldb) target variable -T GlobalF
// CHECK: (const U<int>) GlobalF = {
// CHECK:   (int) K = 10
// CHECK:   (int) L = 11
// CHECK: }
// CHECK: (lldb) target variable -T GlobalG
// CHECK: (const U<int>::V<int>) GlobalG = {
// CHECK:   (int) I = 8
// CHECK:   (int) J = 9
// CHECK: }
// CHECK: (lldb) target modules dump ast
// CHECK: Dumping clang ast for 1 modules.
// CHECK: TranslationUnitDecl {{.*}}
// CHECK: |-CXXRecordDecl {{.*}} struct S definition
// CHECK: | |-FieldDecl {{.*}} C 'int'
// CHECK: | |-FieldDecl {{.*}} D 'int'
// CHECK: | |-FieldDecl {{.*}} DD 'void *'
// CHECK: | `-CXXRecordDecl {{.*}} struct NestedStruct definition
// CHECK: |   |-FieldDecl {{.*}} A 'int'
// CHECK: |   `-FieldDecl {{.*}} B 'int'
// CHECK: |-CXXRecordDecl {{.*}} struct T definition
// CHECK: | |-FieldDecl {{.*}} NT 'int'
// CHECK: | |-CXXRecordDecl {{.*}} struct NestedStruct definition
// CHECK: | | |-FieldDecl {{.*}} E 'int'
// CHECK: | | `-FieldDecl {{.*}} F 'int'
// CHECK: | `-CXXRecordDecl {{.*}} struct U definition
// CHECK: |   |-FieldDecl {{.*}} G 'int'
// CHECK: |   `-FieldDecl {{.*}} H 'int'
// CHECK: |-CXXRecordDecl {{.*}} class U<int> definition
// CHECK: | |-FieldDecl {{.*}} K 'int'
// CHECK: | |-FieldDecl {{.*}} L 'int'
// CHECK: | `-CXXRecordDecl {{.*}} struct W definition
// CHECK: |   |-FieldDecl {{.*}} M 'int'
// CHECK: |   `-FieldDecl {{.*}} N 'int'
// CHECK: |-CXXRecordDecl {{.*}} struct U<int>::V<int> definition
// CHECK: | |-FieldDecl {{.*}} I 'int'
// CHECK: | `-FieldDecl {{.*}} J 'int'
