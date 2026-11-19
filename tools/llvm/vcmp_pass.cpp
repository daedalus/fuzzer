// SIMD trace-cmp: instrument vector `icmp` for the afl_shim.c cmplog layer.
//
// SanitizerCoverage's trace-cmp skips any icmp whose operands are not
// scalar integers, so a SIMD magic check is invisible to cmplog:
//
//   _mm_cmpeq_epi8(v, "ABCD...P")  ->  icmp ne <16 x i8> %v, <65,...,80>
//   trace-cmp logs only cmp2(0, movemask) -- the magic never reaches the log.
//
// This pass runs at OptimizerLastEP (after the vectorizers, like sancov) and
// rewrites nothing; it adds one call ahead of each vector icmp:
//
//   splat constant   icmp eq <32 x i8> %v, splat(0x42)
//                    -> __afl_trace_vcmp_splat(&v, 0x42, 32 lanes, 1 byte)
//   anything else    icmp ne <16 x i8> %a, %b
//                    -> __afl_trace_vcmp(&b, &a, 16 bytes, 1-byte lanes)
//
// Operands are spilled to entry-block allocas so the callback takes plain
// pointers whatever the vector width. Functions carrying
// no_sanitize("coverage") (__AFL_NO_COV, i.e. the shim itself) are skipped.
// Each module gets weak no-op callbacks; the shim's strong ones override.
//
// Build: clang++ $(llvm-config --cxxflags) -shared -fPIC vcmp_pass.cpp
// Use:   clang -fpass-plugin=vcmp_pass.so -fsanitize-coverage=trace-cmp ...

#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

namespace {

constexpr const char *kVcmp = "__afl_trace_vcmp";
constexpr const char *kVcmpSplat = "__afl_trace_vcmp_splat";

// Lane widths the shim's integer record path accepts (bytes).
constexpr unsigned kMaxLaneBytes = 8;

class VcmpPass : public PassInfoMixin<VcmpPass> {
public:
    PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
        bool changed = false;
        for (Function &F : M) changed |= runOn(F);
        return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }

    static bool isRequired() { return true; }

private:
    FunctionCallee vcmp_;
    FunctionCallee splat_;

    void declare(Module &M) {
        LLVMContext &C = M.getContext();
        auto *ptr = PointerType::getUnqual(C);
        auto *i32 = Type::getInt32Ty(C);
        auto *i64 = Type::getInt64Ty(C);
        auto *vd = Type::getVoidTy(C);
        vcmp_ = M.getOrInsertFunction(kVcmp, vd, ptr, ptr, i32, i32);
        splat_ = M.getOrInsertFunction(kVcmpSplat, vd, ptr, i64, i32, i32);
        weakStub(vcmp_);
        weakStub(splat_);
    }

    // Weak hidden no-op, like compiler-rt's sancov stubs: a TU linked
    // without the cmplog shim (zlib_read_tracecmp.so) still links and runs;
    // the shim's strong definition wins wherever it is linked.
    static void weakStub(FunctionCallee cb) {
        auto *F = dyn_cast<Function>(cb.getCallee());
        if (!F || !F->isDeclaration()) return;

        F->setLinkage(GlobalValue::WeakAnyLinkage);
        F->setVisibility(GlobalValue::HiddenVisibility);
        F->addFnAttr(Attribute::NoSanitizeCoverage);
        ReturnInst::Create(F->getContext(), BasicBlock::Create(F->getContext(), "", F));
    }

    // Vector icmp on whole-byte integer lanes, else null.
    static ICmpInst *vectorCmp(Instruction &I) {
        auto *cmp = dyn_cast<ICmpInst>(&I);
        if (!cmp) return nullptr;

        auto *vt = dyn_cast<FixedVectorType>(cmp->getOperand(0)->getType());
        if (!vt) return nullptr;

        auto *lane = dyn_cast<IntegerType>(vt->getElementType());
        if (!lane || lane->getBitWidth() % 8 != 0) return nullptr;
        return cmp;
    }

    bool runOn(Function &F) {
        if (F.isDeclaration() || F.hasFnAttribute(Attribute::NoSanitizeCoverage))
            return false;

        SmallVector<ICmpInst *, 8> sites;
        for (Instruction &I : instructions(F))
            if (ICmpInst *c = vectorCmp(I)) sites.push_back(c);
        if (sites.empty()) return false;

        declare(*F.getParent());
        IRBuilder<> entry(&*F.getEntryBlock().getFirstInsertionPt());
        for (ICmpInst *c : sites) instrument(*c, entry);
        return true;
    }

    // Constant operand side first, so (const, variable) maps to the shim's
    // (a, b) like sancov's trace_const_cmp.
    void instrument(ICmpInst &cmp, IRBuilder<> &entry) {
        Value *a = cmp.getOperand(0), *b = cmp.getOperand(1);
        if (isa<Constant>(a)) std::swap(a, b);

        auto *vt = cast<FixedVectorType>(a->getType());
        unsigned laneBytes = vt->getScalarSizeInBits() / 8;
        IRBuilder<> at(&cmp);
        const DataLayout &DL = cmp.getModule()->getDataLayout();

        auto *k = dyn_cast<Constant>(b);
        Constant *s = k ? k->getSplatValue() : nullptr;
        if (s && isa<ConstantInt>(s) && laneBytes <= kMaxLaneBytes) {
            uint64_t c = cast<ConstantInt>(s)->getZExtValue();
            at.CreateCall(splat_, {spill(a, entry, at), at.getInt64(c),
                                   at.getInt32(vt->getNumElements()), at.getInt32(laneBytes)});
            return;
        }

        unsigned bytes = (unsigned)DL.getTypeStoreSize(vt);
        at.CreateCall(vcmp_, {spill(b, entry, at), spill(a, entry, at), at.getInt32(bytes),
                           at.getInt32(laneBytes)});
    }

    // Entry-block slot, stored at the compare: no dynamic stack growth in loops.
    static Value *spill(Value *v, IRBuilder<> &entry, IRBuilder<> &at) {
        AllocaInst *slot = entry.CreateAlloca(v->getType());
        at.CreateStore(v, slot);
        return slot;
    }
};

}  // namespace

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
    return {LLVM_PLUGIN_API_VERSION, "afl-vcmp", "1", [](PassBuilder &PB) {
                PB.registerOptimizerLastEPCallback(
                    [](ModulePassManager &MPM, OptimizationLevel) { MPM.addPass(VcmpPass()); });
            }};
}
