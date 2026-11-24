# SPDX-License-Identifier: Apache-2.0
"""Compile the port's actual device function bodies as a serial host model.

This checks C++ algorithm semantics, not HIP codegen, parallel races, DMA, memory
coherency, wave execution, Graphs or model accuracy. GPU tests remain separate.
"""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _host_model_namespace(source):
    namespace = source[source.index('constexpr int kWarpSize'):source.index('int64_t check_2d_rows')]
    # Keep the pure attention-range helper and device bodies, but exclude
    # Torch/HIP host validation. Function anchors survive registration-comment
    # changes and do not replace the native span/alias/device checks with mocks.
    start = namespace.index('void check_host_query(')
    end = namespace.index('__device__ __forceinline__ int32_t hash_slot')
    return namespace[:start] + namespace[end:]


def test_host_model_extraction_keeps_pure_helpers_without_host_runtime():
    source = (ROOT / 'csrc/hisparse/hisparse_cpp.cu').read_text(encoding='utf-8')
    namespace = _host_model_namespace(source)
    assert 'bool attention_range_fits(' in namespace
    for function in ('hash_slot', 'copy_row_warp', 'zero_row_warp',
                     'hisparse_resolve_residency_kernel',
                     'hisparse_gather_plan_kernel', 'hisparse_gather_compact_kernel',
                     'hisparse_invalidate_written_slots_kernel'):
        assert function + '(' in namespace
    for host_only in ('at::Tensor', 'TORCH_CHECK', 'hipError_t',
                      'mapped_host_pointer(', 'void check_host_query('):
        assert host_only not in namespace


def test_native_metadata_and_gathers_on_cpu(tmp_path):
    compiler = shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')
    if compiler is None:
        pytest.skip('host C++ compiler unavailable; not a HIP/device test')
    source = (ROOT / 'csrc/hisparse/hisparse_cpp.cu').read_text(encoding='utf-8')
    namespace = _host_model_namespace(source)
    harness = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <climits>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>
using std::min; using std::max;
#define __device__
#define __global__
#define __forceinline__ inline
#define __launch_bounds__(...)
#define __shared__
#define __syncthreads() ((void)0)
struct Dim { unsigned x=0,y=0,z=0; };
Dim threadIdx,blockIdx,gridDim,blockDim;
struct alignas(16) uint4 { unsigned x,y,z,w; };
inline unsigned __umulhi(unsigned a,unsigned b) {return (uint64_t(a)*b)>>32;}
template<class T,class U> T atomicAdd(T* p,U v) {T old=*p;*p+=v;return old;}
inline int atomicCAS(int* p,int old,int value) {int before=*p;if(before==old)*p=value;return before;}
inline int atomicMin(int* p,int value) {int old=*p;*p=std::min(old,value);return old;}
alignas(16) char smem_raw[300000];
'''
    harness += namespace
    harness += r'''
void resolve(int rows,int topk,int* indices,int* dgi,int16_t* lru,int* out,
             int* mm,int* att,int* resolved,int* valid,int* swaps_h,int* swaps_d,
             int* counts,unsigned long long* stats,int* requests=nullptr,
             int* source=nullptr,int* resident=nullptr,int* state=nullptr) {
  int hot_table[]={1,2};
  blockDim.x=1;gridDim.x=rows;threadIdx.x=0;
  for(unsigned row=0;row<unsigned(rows);++row) {
    blockIdx.x=row;
    // Emulate the parallel initialization of all five counters.
    std::memset(smem_raw,0,sizeof(smem_raw));
    hisparse_resolve_residency_kernel(hot_table,indices,requests,source,resident,
        resolved,valid,swaps_h,swaps_d,counts,out,att,mm,dgi,lru,stats,state,
        state?1:0,1,12,1,64,2,2,topk,4,5,4,5,
        2,source?1:0,source?2:0,source?2:0,
        2,resident?1:0,resident?2:0,resident?2:0,0,topk,topk,1);
  }
}
int main() {
  assert(attention_range_fits(12, 2, 5));
  assert(attention_range_fits(2, 2, INT64_MAX)); // one page never multiplies stride
  assert(attention_range_fits(4, 2, INT32_MAX - 1));
  assert(!attention_range_fits(4, 2, INT32_MAX));
  assert(!attention_range_fits(12, 2, INT64_MAX));
  assert(!attention_range_fits(12, 2, 1));
  assert(!attention_range_fits(12, 0, 5));

  { // hits, unique misses, duplicate references, holes, physical/attention pitch.
    int indices[]={12,20,20,-1},dgi[]={10,11,12,13};int16_t lru[]={0,1,2,3};
    int out[4],mm[4],att[4],res[4],valid[1],sh[4],sd[4],counts[1];
    unsigned long long stats[2]={0,0};
    resolve(1,4,indices,dgi,lru,out,mm,att,res,valid,sh,sd,counts,stats);
    assert(out[0]==4 && out[1]==2 && out[2]==2 && out[3]==-1);
    assert(att[0]==10 && att[1]==5 && att[2]==5 && att[3]==-1);
    assert(mm[0]==0 && mm[1]==1 && mm[2]==0 && mm[3]==0);
    assert(valid[0]==3 && counts[0]==1 && sh[0]==20 && sd[0]==2);
    assert(stats[0]==2 && stats[1]==1);
    assert(dgi[0]==20 && lru[0]==1 && lru[1]==3 && lru[2]==0 && lru[3]==2);
    int changed[]={10,11,20,-1};
    resolve(1,4,changed,dgi,lru,out,mm,att,res,valid,sh,sd,counts,stats);
    assert(out[0]==5 && out[1]==3 && out[2]==2 && counts[0]==1);
    assert(lru[0]==2 && lru[1]==3 && lru[2]==1 && lru[3]==0);
  }
  { // #69057: stable valid-front packing with holes after LRU resolution.
    int indices[]={12,-1,20,20},dgi[]={10,11,12,13};int16_t lru[]={0,1,2,3};
    int out[4],mm[4],att[4],res[4],valid[1],sh[4],sd[4],counts[1];
    unsigned long long stats[2]={0,0};
    resolve(1,4,indices,dgi,lru,out,mm,att,res,valid,sh,sd,counts,stats);
    assert(out[0]==4 && out[1]==-1 && out[2]==2 && out[3]==2);
    assert(att[0]==10 && att[1]==5 && att[2]==5 && att[3]==-1 && valid[0]==3);
    assert(mm[0]==0 && mm[1]==0 && mm[2]==1 && mm[3]==0);
  }
  { // Multiple contiguous rows share one union and compact swap segment.
    int indices[]={0,4,0,5},source[]={5,7},requests[]={0,0};
    int dgi[]={10,11,12,13};int16_t lru[]={0,1,2,3};
    int out[4],mm[4],att[4],res[4],valid[2],sh[4],sd[4],counts[2];
    unsigned long long stats[2]={0,0};
    // Source token4/5 lies beyond the two-column table and is invalid.
    resolve(2,2,indices,dgi,lru,out,mm,att,res,valid,sh,sd,counts,stats,requests,source);
    assert(out[0]==2 && out[1]==-1 && out[2]==2 && out[3]==-1);
    assert(counts[0]==0 && counts[1]==0 && valid[0]==1 && valid[1]==1);
    assert(stats[0]==2 && stats[1]==0);
    int changed[]={0,2,0,3}; // global10/14/10/15: 2 unique misses.
    resolve(2,2,changed,dgi,lru,out,mm,att,res,valid,sh,sd,counts,stats,requests,source);
    assert(out[0]==2 && out[1]==3 && out[2]==2 && out[3]==4);
    assert(sh[0]==14 && sh[1]==15 && sd[0]==3 && sd[1]==4);
    assert(counts[0]==2 && counts[1]==0 && mm[1]==1 && mm[3]==1);
  }
  { // Resident-only fast path leaves hot ownership and LRU untouched.
    int indices[]={0,1},source[]={3,4},resident[]={1,0},requests[]={0},state[]={0};
    int dgi[]={10,11,12,13};int16_t lru[]={0,1,2,3};
    int out[2],mm[2],att[2],res[2],valid[1],sh[2],sd[2],counts[1];
    unsigned long long stats[2]={0,0};
    resolve(1,2,indices,dgi,lru,out,mm,att,res,valid,sh,sd,counts,stats,requests,source,resident,state);
    assert(out[0]==2 && out[1]==3 && res[0]==6 && res[1]==7);
    assert(counts[0]==0 && valid[0]==2 && dgi[0]==10 && lru[0]==0);
    assert(stats[0]==0 && stats[1]==0);
    int holes[]={-1,1};
    resolve(1,2,holes,dgi,lru,out,mm,att,res,valid,sh,sd,counts,stats,requests,source,resident,state);
    assert(out[0]==-1 && out[1]==3 && att[0]==6 && att[1]==-1 && valid[0]==1);
    state[0]=-1;mm[0]=mm[1]=99;
    resolve(1,2,indices,dgi,lru,out,mm,att,res,valid,sh,sd,counts,stats,requests,source,resident,state);
    assert(out[0]==-1 && out[1]==-1 && valid[0]==0 && counts[0]==0);
    assert(mm[0]==0 && mm[1]==0);
  }
  { // Actual byte-copy kernels, including gaps, out-of-host zero-fill, and masks.
    char host[8*16],hot[6*40];
    for(int i=0;i<128;++i)host[i]=char(i);
    std::memset(hot,0x55,sizeof(hot));
    int g[]={2,50,-1},dst[]={2,3,-1},mm[]={1,1,1},att[3];
    blockDim.x=32;gridDim.x=gridDim.y=1;blockIdx.x=blockIdx.y=0;
    for(threadIdx.x=0;threadIdx.x<32;++threadIdx.x)
      hisparse_gather_plan_kernel(host,hot,g,dst,mm,att,nullptr,8,12,16,40,2,3,5);
    assert(std::memcmp(hot+40,host+32,16)==0);
    for(int i=56;i<72;++i)assert(hot[i]==0);
    assert(hot[72]==0x55 && hot[39]==0x55 && att[0]==5 && att[1]==6 && att[2]==-1);
    std::memset(hot,0x55,sizeof(hot));int counts[]={1};
    for(threadIdx.x=0;threadIdx.x<32;++threadIdx.x)
      hisparse_gather_compact_kernel(host,hot,g,dst,counts,8,12,16,40,2,3);
    assert(std::memcmp(hot+40,host+32,16)==0 && hot[56]==0x55);
    int state[]={-1};
    for(threadIdx.x=0;threadIdx.x<32;++threadIdx.x)
      hisparse_gather_plan_kernel(host,hot,g,dst,mm,att,state,8,12,16,40,2,3,5);
    assert(att[0]==-1 && att[1]==-1 && att[2]==-1);
  }
  { // Invalidate every duplicate slot for the requested state; ignore padding.
    int dgi[]={7,9,7,11},state[]={0},requests[]={0,-1};int64_t written[]={7,9};
    blockDim.x=4;gridDim.x=2;
    for(blockIdx.x=0;blockIdx.x<2;++blockIdx.x)
      for(threadIdx.x=0;threadIdx.x<4;++threadIdx.x)
        hisparse_invalidate_written_slots_kernel(dgi,state,requests,written,2,1,1,4);
    assert(dgi[0]==-1 && dgi[1]==9 && dgi[2]==-1 && dgi[3]==11);
  }

  { // A source row straddles two NON-adjacent aliases at an unaligned byte.
    char left[35],right[93],hot[6*40];
    for(int i=0;i<35;++i)left[i]=char(i);
    for(int i=0;i<93;++i)right[i]=char(i+35);
    std::memset(hot,0x55,sizeof(hot));
    int g[]={2},dst[]={2},mm[]={1},att[1];
    blockDim.x=32;gridDim.x=gridDim.y=1;blockIdx.x=blockIdx.y=0;
    for(threadIdx.x=0;threadIdx.x<32;++threadIdx.x)
      hisparse_gather_plan_kernel(left,hot,g,dst,mm,att,nullptr,8,12,16,40,2,1,5,0,35);
    for(threadIdx.x=0;threadIdx.x<32;++threadIdx.x)
      hisparse_gather_plan_kernel(right,hot,g,dst,mm,att,nullptr,8,12,16,40,2,1,5,35,128);
    for(int i=0;i<16;++i)assert(hot[40+i]==char(32+i));
    assert(hot[39]==0x55 && hot[56]==0x55);
  }

  for (int physical_block : {64, 256}) {
    int table[]={1},indices[]={63,-1,64,64,65},request[]={0},source[]={1,2},state[]={0};
    int dgi[64];int16_t lru[64];
    for(int i=0;i<64;++i){dgi[i]=256+i;lru[i]=i;}
    int out[5],mask[5],attention[5],resolved[5],valid[1],sh[5],sd[5],counts[1];
    std::fill(sh,sh+5,-77);std::fill(sd,sd+5,-77);
    unsigned long long stats[2]={0,0};
    blockDim.x=1;gridDim.x=1;threadIdx.x=blockIdx.x=0;
    std::memset(smem_raw,0,sizeof(smem_raw));
    hisparse_resolve_residency_kernel(table,indices,request,source,nullptr,
        resolved,valid,sh,sd,counts,out,attention,mask,dgi,lru,stats,state,
        1,1,4*physical_block,1,1024,1,physical_block,5,64,65,64,256,
        2,1,2,256,0,0,0,0,0,5,5,1);
    int want_out[]={physical_block+63,-1,physical_block,physical_block,physical_block+1};
    int want_attention[]={319,256,256,257,-1},want_resolved[]={319,-1,320,320,321};
    assert(std::equal(out,out+5,want_out));
    assert(std::equal(attention,attention+5,want_attention));
    assert(std::equal(resolved,resolved+5,want_resolved));
    assert(valid[0]==4 && counts[0]==2 && sh[0]==320 && sh[1]==321);
    assert(sd[0]==physical_block && sd[1]==physical_block+1);
    assert(dgi[0]==320 && dgi[1]==321 && stats[0]==2 && stats[1]==2);
  }
  for(int block : {64,128,256}) {
    constexpr int width=16;const int pitch=block*width+24,offset=3;
    std::vector<char> host(8*width),actual(offset+4*pitch+17,0x55);
    for(int i=0;i<8*width;++i)host[i]=char(i%113);
    int globals[]={0,1,2,99,3,4,5};
    int dst[]={block-1,block,block+min(63,block-2),2*block-1,2*block,2*block+64,3*block-1};
    for(int i=0;i<7;++i)for(int j=i+1;j<7;++j)assert(dst[i]!=dst[j]);
    int mask[]={1,1,1,1,1,1,1},count[]={7},attention[7];
    std::vector<char> expected=actual;
    for(int i=0;i<7;++i){
      auto* p=expected.data()+offset+(dst[i]/block)*pitch+(dst[i]%block)*width;
      if(globals[i]<8)std::memcpy(p,host.data()+globals[i]*width,width);
      else std::memset(p,0,width);
    }
    for(int kind=0;kind<2;++kind){
      std::fill(actual.begin(),actual.end(),0x55);
      blockDim.x=32;gridDim.x=gridDim.y=1;blockIdx.x=blockIdx.y=0;
      for(threadIdx.x=0;threadIdx.x<32;++threadIdx.x){
        if(kind==0)hisparse_gather_plan_kernel(host.data(),actual.data()+offset,
            globals,dst,mask,attention,nullptr,8,4*block,width,pitch,block,7,256);
        else hisparse_gather_compact_kernel(host.data(),actual.data()+offset,
            globals,dst,count,8,4*block,width,pitch,block,7);
      }
      assert(actual==expected);
      if(kind==0)for(int i=0;i<7;++i)
        assert(attention[i]==(dst[i]/block)*256+dst[i]%block);
    }
  }

  std::cout << "serial host model: resolve/replay/resident/padding/plan/compact/invalidate passed\n";
}
'''
    cpp = tmp_path / 'hisparse_host.cpp'
    binary = tmp_path / 'hisparse_host'
    cpp.write_text(harness, encoding='utf-8')
    p = subprocess.run([compiler, '-std=c++17', '-O1', '-Wall', '-Wextra',
                        '-fsanitize=address,undefined', str(cpp), '-o', str(binary)],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    p = subprocess.run([str(binary)], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert 'passed' in p.stdout
