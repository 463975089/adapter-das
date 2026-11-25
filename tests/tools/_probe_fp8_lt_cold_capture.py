"""Fresh-process regression: tuned eager first, unseen native shape in capture."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from vllm.config import CompilationConfig,VllmConfig,set_current_vllm_config
from vllm.plugins import load_general_plugins
R=Path(__file__).resolve().parent
@torch.inference_mode()
def main():
 load_general_plugins();torch.cuda.set_device(0);torch.set_num_threads(4)
 from hcu_adapter.bridges.vllm_scaled_mm import install
 from hcu_adapter.kernels.fp8_scaled_mm_cpp import _library
 from vllm.model_executor.kernels.linear.scaled_mm import triton as core
 config=VllmConfig(compilation_config=CompilationConfig(mode=0,custom_ops=['all']))
 config.model_config=SimpleNamespace(dtype=torch.bfloat16)
 with set_current_vllm_config(config):
  install()
  a=torch.randn(42,1024,device='cuda').to(torch.float8_e4m3fn)
  b=torch.randn(1024,1024,device='cuda').to(a.dtype).T
  sa=torch.rand(42,1,device='cuda')*.1+.001;sb=torch.rand(1024,1,device='cuda')*.1+.001
  def call(x,s):return core.triton_scaled_mm(x,b,s,sb,torch.bfloat16,None)
  # Do not call the control provider: that would hide its lazy-handle bug.
  for _ in range(3):call(a,sa)
  assert _library.cache_info().currsize==1
  x=torch.randn(257,1024,device='cuda').to(a.dtype);sx=torch.rand(257,1,device='cuda')*.1+.001
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):y=call(x,sx)
  for _ in range(3):
   x.copy_(torch.randn_like(x,dtype=torch.float32).to(x.dtype));sx.uniform_(.001,.1)
   graph.replay();expected=call(x,sx);torch.cuda.synchronize()
   assert torch.equal(y.view(torch.uint8),expected.view(torch.uint8))
  print('cold vendor capture: 3 changed-input replays passed',flush=True)
if __name__=='__main__':main()
