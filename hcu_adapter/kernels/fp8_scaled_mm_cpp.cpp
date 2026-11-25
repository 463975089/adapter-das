// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
// Framework-free, zero-workspace hipBLASLt token/channel GEMM.
#include <hipblaslt/hipblaslt-ext.hpp>
#include <algorithm>
#include <array>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {
void check(hipblasStatus_t status) {
    if(status!=HIPBLAS_STATUS_SUCCESS)throw std::runtime_error("hipBLASLt failure");
}
void hip_check(hipError_t status) {
    if(status!=hipSuccess)throw std::runtime_error("HIP failure");
}
struct State {
    hipblasLtHandle_t handle=nullptr;
    hipblasLtMatmulDesc_t desc=nullptr;
    hipblasLtMatmulPreference_t pref=nullptr;
    std::array<hipblasLtMatrixLayout_t,4> layouts{};
    hipblasLtMatmulAlgo_t algo{};
    bool ready=false,rejected=false;
    State(int m,int n,int k) {
      try {
        check(hipblasLtCreate(&handle));
        check(hipblasLtMatmulDescCreate(&desc,HIPBLAS_COMPUTE_32F,HIP_R_32F));
        hipblasOperation_t ta=HIPBLAS_OP_T,tb=HIPBLAS_OP_N;
        check(hipblasLtMatmulDescSetAttribute(desc,HIPBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));
        check(hipblasLtMatmulDescSetAttribute(desc,HIPBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb)));
        auto mode=HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
        check(hipblasLtMatmulDescSetAttribute(desc,HIPBLASLT_MATMUL_DESC_A_SCALE_MODE,&mode,sizeof(mode)));
        check(hipblasLtMatmulDescSetAttribute(desc,HIPBLASLT_MATMUL_DESC_B_SCALE_MODE,&mode,sizeof(mode)));
        check(hipblasLtMatrixLayoutCreate(&layouts[0],HIP_R_8F_E4M3,k,n,k));
        check(hipblasLtMatrixLayoutCreate(&layouts[1],HIP_R_8F_E4M3,k,m,k));
        check(hipblasLtMatrixLayoutCreate(&layouts[2],HIP_R_16BF,n,m,n));
        check(hipblasLtMatrixLayoutCreate(&layouts[3],HIP_R_16BF,n,m,n));
        check(hipblasLtMatmulPreferenceCreate(&pref));
        size_t workspace=0;
        check(hipblasLtMatmulPreferenceSetAttribute(pref,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace,sizeof(workspace)));
      } catch(...) {release();throw;}
    }
    ~State() {release();}
    void release() {
        if(pref)hipblasLtMatmulPreferenceDestroy(pref);
        for(auto layout:layouts)if(layout)hipblasLtMatrixLayoutDestroy(layout);
        if(desc)hipblasLtMatmulDescDestroy(desc);
        if(handle)hipblasLtDestroy(handle);
    }
    void scales(const void* sa,const void* sw) {
        check(hipblasLtMatmulDescSetAttribute(desc,HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,&sw,sizeof(sw)));
        check(hipblasLtMatmulDescSetAttribute(desc,HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,&sa,sizeof(sa)));
    }
    void run(const hipblasLtMatmulAlgo_t& selected,const void* a,const void* w,void* out,hipStream_t stream) {
        const float alpha=1,beta=0;
        check(hipblasLtMatmul(handle,desc,&alpha,w,layouts[0],a,layouts[1],&beta,
            out,layouts[2],out,layouts[3],&selected,nullptr,0,stream));
    }
    void tune(const void* a,const void* w,void* out,hipStream_t stream) {
        std::array<hipblasLtMatmulHeuristicResult_t,256> choices{};int count=0;
        check(hipblasLtMatmulAlgoGetHeuristic(handle,desc,layouts[0],layouts[1],
            layouts[2],layouts[3],pref,choices.size(),choices.data(),&count));
        if(count==0 || choices[0].state!=HIPBLAS_STATUS_SUCCESS || choices[0].workspaceSize)return;
        hipEvent_t begin=nullptr,end=nullptr;
        hip_check(hipEventCreate(&begin));hip_check(hipEventCreate(&end));
        try {
            auto time=[&](const hipblasLtMatmulAlgo_t& candidate) {
                for(int i=0;i<3;++i)run(candidate,a,w,out,stream);
                std::array<float,3> times{};
                for(float& elapsed:times) {
                    hip_check(hipEventRecord(begin,stream));
                    for(int i=0;i<10;++i)run(candidate,a,w,out,stream);
                    hip_check(hipEventRecord(end,stream));hip_check(hipEventSynchronize(end));
                    hip_check(hipEventElapsedTime(&elapsed,begin,end));
                }
                std::sort(times.begin(),times.end());return times[1];
            };
            algo=choices[0].algo;float best=time(algo);const float original=best;
            for(int i=1;i<count;++i) {
                auto& candidate=choices[i];
                if(candidate.state!=HIPBLAS_STATUS_SUCCESS || candidate.workspaceSize)continue;
                const auto name=hipblaslt_ext::getSolutionNameFromAlgo(handle,candidate.algo);
                // Qualify the installed deterministic channelwise family, rather
                // than storing version-specific algorithm indices or one shape.
                if(name.find("Cijk_Alik_Bljk_F8BS_MT")!=0 ||
                   name.find("_bias_channelwise")==std::string::npos ||
                   (name.find("_GSU")!=std::string::npos && name.find("_GSU1_SU0_")==std::string::npos))continue;
                const float elapsed=time(candidate.algo);
                if(elapsed<best*.98f && elapsed<original*.95f) {best=elapsed;algo=candidate.algo;}
            }
            ready=true;
        } catch(...) {
            hipEventDestroy(begin);hipEventDestroy(end);throw;
        }
        hipEventDestroy(begin);hipEventDestroy(end);
    }
};
// Descriptors own no tensors or device scratch. A separate cache for each host
// thread and device avoids descriptor mutation across concurrent host calls.
// Streams can share the host descriptor: matmul copies kernel parameters when
// enqueued, and there is no shared device workspace or cached tensor pointer.
using Key=std::tuple<int,int,int,int>;
thread_local std::map<Key,std::unique_ptr<State>> cache;
}

// 0=fallback without touching output; 1=executed; negative=execution failure.
extern "C" int hcu_fp8_lt_run(int m,int n,int k,const void* a,const void* w,
        const void* sa,const void* sw,void* out,void* raw_stream,int allow_tune) {
    try {
        int device;hip_check(hipGetDevice(&device));
        auto stream=static_cast<hipStream_t>(raw_stream);
        Key key{device,m,n,k};
        auto entry=cache.find(key);
        if(entry==cache.end()) {
            if(!allow_tune || cache.size()>=128)return 0;
            entry=cache.emplace(key,std::make_unique<State>(m,n,k)).first;
        }
        State& state=*entry->second;state.scales(sa,sw);
        if(state.rejected)return 0;
        if(!state.ready) {
            if(!allow_tune)return 0;
            state.tune(a,w,out,stream);
            if(!state.ready) {state.rejected=true;return 0;}
        }
        state.run(state.algo,a,w,out,stream);return 1;
    } catch(...) {return -1;}
}

extern "C" int hcu_fp8_lt_name(int device,int m,int n,int k,char* output,size_t size) {
    try {
        auto entry=cache.find(Key{device,m,n,k});
        if(entry==cache.end() || !entry->second->ready)return 0;
        State& state=*entry->second;
        auto name=hipblaslt_ext::getSolutionNameFromAlgo(state.handle,state.algo);
        if(name.size()+1>size)return -1;
        std::memcpy(output,name.c_str(),name.size()+1);return 1;
    } catch(...) {return -1;}
}
