import time
import functools
import torch
import jax
import jax.numpy as jnp
from torch.utils.cpp_extension import load


from myconv import ConvModel
from myconv_jax import conv2d_manual_jax


import matplotlib.pyplot as plt

N, C = 4, 16
runs = 20

# kinda similar to the TIME macro in MP1
def TIME(name, fn):
    fn()

    total = 0.0
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        total += time.perf_counter() - start
    ms = total / runs * 1000

    print(f"Time taken for conv {name}: {ms} ms")
    return ms

t = time.perf_counter()

cuda_mod = load(name="myconv", sources=["myconv_kernel.cu"])
print(f"CUDA build time: {time.perf_counter() - t:.1f} s")


def run(H, K):
    pad = K // 2

    # build input, model, and weights
    x = torch.randn(N, C, H, H, device="cuda")
    model = ConvModel(H, H, C, C, K, 1, pad).cuda().eval()
    w = model.weight.detach()

    # name -> time
    times = {}

    with torch.no_grad():
        times["interpreter"] = TIME("interpreter", lambda: (model(x), torch.cuda.synchronize()))
        torch._dynamo.reset()

        # test compile time
        compiled = torch.compile(model, backend="inductor")
        t = time.perf_counter()
        compiled(x)
        torch.cuda.synchronize()
        print(f"H={H} K={K} inductor compile: {time.perf_counter() - t:.1f} s")

        # get running times
        times["inductor"] = TIME("inductor", lambda: (compiled(x), torch.cuda.synchronize()))
        times["cuda"] = TIME("cuda", lambda: (cuda_mod.conv_cuda(x, w, 1, pad), torch.cuda.synchronize()))

    # build jax versions of all the bits
    xj, wj, bj = jnp.array(x.cpu()), jnp.array(w.cpu()), jnp.array(model.bias.detach().cpu())
    f = jax.jit(functools.partial(conv2d_manual_jax, stride=1, padding=pad))

    # test compile time
    t = time.perf_counter()
    f(xj, wj, bj).block_until_ready()
    print(f"H={H} K={K} jax compile: {time.perf_counter() - t:.1f} s")

    # get running time
    times["jax"] = TIME("jax", lambda: f(xj, wj, bj).block_until_ready())
    return times


def sweep(name, sizes, l, xlabel):
    results = {}
    for s in sizes:
        results[s] = run(*l(s))
        print(name, s, results[s])

    # plotting cause sheets is lowk ugly
    plt.figure()
    for backend in ["interpreter", "inductor", "jax", "cuda"]:
        plt.plot(sizes, [results[s][backend] for s in sizes], marker="o", label=backend)
    plt.xlabel(xlabel)
    plt.ylabel("ms per call")
    plt.xscale("log", base=2); plt.yscale("log")
    plt.legend()
    plt.savefig(f"sweep_{name}.png", dpi=150)


sweep("input", [16, 32, 64, 128, 256], lambda h: (h, 3), "input size H=W (K=3)")
sweep("filter", [1, 3, 5, 7, 9], lambda k: (64, k), "filter size K (H=W=64)")
