"""ctypes wrapper for native packed FP4 GEMM. Run only inside a GPU allocation."""
import ctypes as C
import torch

def ptr(x):return None if x is None else x.data_ptr()
def stream():return torch.cuda.current_stream().cuda_stream

class Runtime:
 def __init__(self,path):
  self.lib=C.CDLL(str(path));v=C.c_void_p;i=C.c_int
  self.lib.mf_error.restype=C.c_char_p
  self.lib.mf_sf_size.argtypes=[i,i];self.lib.mf_sf_size.restype=C.c_size_t
  self.lib.mf_swizzle.argtypes=[v,v,i,i,v];self.lib.mf_swizzle.restype=i
  self.lib.mf_create.argtypes=[i,i,i,v,v,v,v,v,v,C.c_float,v];self.lib.mf_create.restype=v
  self.lib.mf_run.argtypes=[v,v,v,i];self.lib.mf_run.restype=i
  self.lib.mf_dump.argtypes=[v,v,v,v,v];self.lib.mf_dump.restype=i
  self.lib.mf_destroy.argtypes=[v];self.lib.mf_destroy.restype=None
 def check(self,value):
  if value:raise RuntimeError(self.lib.mf_error().decode())
 def scales(self,s):
  n,g=s.shape;k=g*16
  out=torch.empty(self.lib.mf_sf_size(n,k),device='cuda',dtype=torch.uint8)
  self.check(self.lib.mf_swizzle(ptr(s),ptr(out),n,k,stream()));return out

@torch.no_grad()
def encode(w,mask=None):
 """Exact FourOverSix/E0M3 code and scale construction, independent of CUDA producer."""
 n,k=w.shape;global_scale=(w.float().abs().amax()/(6*448)).clamp_min(torch.finfo(torch.float32).tiny)
 packed=torch.empty((n,k//2),device=w.device,dtype=torch.uint8);scales=torch.empty((n,k//16),device=w.device,dtype=torch.uint8)
 lut=torch.tensor([0,.5,1,1.5,2,3,4,6],device=w.device);thresholds=torch.tensor([.25,.75,1.25,1.75,2.5,3.5,5],device=w.device)
 for start in range(0,n,256):
  v=w[start:start+256].float().reshape(-1,16)/global_scale;mx=v.abs().amax(-1,keepdim=True)
  s6=(mx/6).clamp(2**-9,448).to(torch.float8_e4m3fn).float();s4=(mx/4).clamp(2**-9,448).to(torch.float8_e4m3fn).float()
  c6=torch.bucketize((v/s6).abs().contiguous(),thresholds);c4=torch.bucketize((v/s4).abs().contiguous(),thresholds)
  sign=v.sign();e6=(lut[c6]*sign*s6-v).square().sum(-1,keepdim=True);e4=(lut[c4]*sign*s4-v).square().sum(-1,keepdim=True)
  use=e4<e6;s=torch.where(use,s4,s6);c=torch.where(use,c4,c6).to(torch.uint8)|((v<0).to(torch.uint8)*8)
  if mask is not None:
   tile=mask[start//256].repeat_interleave(4).repeat(v.shape[0]//(k//16)).reshape(-1,1)
   s0=(mx*(1/7)).clamp(2**-9,448).to(torch.float8_e4m3fn).float();q=(v/s0).round().clamp(-7,7)
   c0=q.abs().to(torch.uint8)|((q<0).to(torch.uint8)*8)
   s=torch.where(tile,s0,s);c=torch.where(tile,c0,c)
  c=c.reshape(-1,k);packed[start:start+c.shape[0]]=(c[:,::2]|(c[:,1::2]<<4))
  scales[start:start+c.shape[0]]=s.to(torch.float8_e4m3fn).view(torch.uint8).reshape(-1,k//16)
 return packed,scales,float(global_scale)

@torch.no_grad()
def decode(packed,scales,global_scale,mask=None):
 n,half=packed.shape;k=half*2
 codes=torch.stack((packed&15,packed>>4),-1).reshape(n,k)
 lut=torch.tensor([0,.5,1,1.5,2,3,4,6],device=packed.device)
 val=lut[(codes&7).long()]
 if mask is not None:
  full=mask.repeat_interleave(256,0).repeat_interleave(64,1)[:n,:k]
  val=torch.where(full,(codes&7).float(),val)
 val=val*torch.where((codes&8)!=0,-1.,1.)
 return val*scales.view(torch.float8_e4m3fn).float().repeat_interleave(16,1)*global_scale

class Linear(torch.nn.Module):
 def __init__(self,runtime,w,mask=None,row_perm=None,col_perm=None):
  super().__init__();self.runtime=runtime;self.n,self.k=w.shape;self.plans={};self.rows=None;self.cols=None;self.mask=None
  if row_perm is not None:
   row_perm=row_perm.to(w.device).long();col_perm=col_perm.to(w.device).long()
   assert torch.equal(torch.sort(row_perm).values,torch.arange(self.n,device=w.device))
   assert torch.equal(torch.sort(col_perm).values,torch.arange(self.k,device=w.device))
   assert torch.equal(col_perm.reshape(-1,16)%16,torch.arange(16,device=w.device).expand(self.k//16,16))
   if not torch.equal(row_perm,torch.arange(self.n,device=w.device)):self.rows=torch.argsort(row_perm).int()
   if not torch.equal(col_perm,torch.arange(self.k,device=w.device)):self.cols=torch.argsort(col_perm.reshape(-1,16)[:,0]//16).int()
   w=w[row_perm][:,col_perm].contiguous()
  if mask is not None and mask.any():self.mask=mask.to(device=w.device,dtype=torch.uint8).contiguous()
  self.packed,self.flat_scales,self.global_scale=encode(w,None if self.mask is None else self.mask.bool())
  self.swizzled=runtime.scales(self.flat_scales)
 def prepare(self,m):
  if m not in self.plans:
   y=torch.empty((m,self.n),device='cuda',dtype=torch.bfloat16)
   p=self.runtime.lib.mf_create(m,self.n,self.k,ptr(self.packed),ptr(self.swizzled),ptr(self.mask),ptr(self.cols),ptr(self.rows),ptr(y),self.global_scale,stream())
   if not p:raise RuntimeError(self.runtime.lib.mf_error().decode())
   self.plans[m]=(p,y)
  return self.plans[m]
 def forward(self,x):
  assert x.dtype==torch.bfloat16 and x.is_contiguous() and x.shape[-1]==self.k
  shape=x.shape[:-1];p,y=self.prepare(x.numel()//self.k)
  self.runtime.check(self.runtime.lib.mf_run(p,ptr(x),stream(),0));return y.view(*shape,self.n)
 def close(self):
  for p,_ in self.plans.values():self.runtime.lib.mf_destroy(p)
  self.plans.clear()
 def activation(self,m):
  p,_=self.prepare(m);codes=torch.empty((m,self.k//2),device='cuda',dtype=torch.uint8);scales=torch.empty((m,self.k//16),device='cuda',dtype=torch.uint8);mx=torch.empty(1,device='cuda',dtype=torch.float32)
  self.runtime.check(self.runtime.lib.mf_dump(p,ptr(codes),ptr(scales),ptr(mx),stream()))
  return codes,scales,float(mx)/(6*448)
