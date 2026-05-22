import torch, sys, traceback, signal
import aiter
from aiter.ops.shuffle import shuffle_weight

device = "cuda"
torch.manual_seed(0)


def handler(sig, frame):
    print("  TIMEOUT after 180s - likely GPU hang", flush=True)
    sys.exit(124)


def run(M, N, K, label):
    print(f"\n--- {label}: M={M} N={N} K={K} (K%64={K%64}, K%128={K%128}) ---", flush=True)
    A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    W = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    A_fp8 = A.to(torch.float8_e4m3fn)
    W_fp8 = W.to(torch.float8_e4m3fn)
    a_scale = torch.ones(M, 1, dtype=torch.float32, device=device)
    w_scale = torch.ones(N, 1, dtype=torch.float32, device=device)
    print(f"  W_fp8 (N,K): {W_fp8.shape}", flush=True)
    try:
        W_shuf = shuffle_weight(W_fp8, layout=(16, 16))
        print(f"  W_shuf: {W_shuf.shape}", flush=True)
        if hasattr(W_shuf, "aiter_padded_k"):
            print(f"  AUTO-PADDED: original_k={W_shuf.aiter_original_k} padded_k={W_shuf.aiter_padded_k}", flush=True)
    except Exception as e:
        print(f"  SHUFFLE EXC: {type(e).__name__}: {str(e)[:300]}", flush=True)
        return
    try:
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(180)
        from aiter.ops.gemm_op_a8w8 import gemm_a8w8_bpreshuffle
        Y = gemm_a8w8_bpreshuffle(A_fp8, W_shuf, a_scale, w_scale, None, torch.bfloat16)
        torch.cuda.synchronize()
        signal.alarm(0)
        has_nan = torch.isnan(Y).any().item()
        has_inf = torch.isinf(Y).any().item()
        Y_ref = (A.float() @ W.float().T).to(torch.bfloat16)
        max_err = (Y - Y_ref).abs().max().item()
        rel_err = (Y - Y_ref).abs().mean().item() / max(Y_ref.abs().mean().item(), 1e-6)
        verdict = "FAIL" if (max_err > 200 or has_nan or has_inf) else "PASS"
        print(f"  Y shape: {Y.shape}, NaN: {has_nan}, Inf: {has_inf}", flush=True)
        print(f"  Max err: {max_err:.4f}, Rel err: {rel_err:.4f}  ===> {verdict}", flush=True)
    except Exception as e:
        signal.alarm(0)
        print(f"  EXCEPTION: {type(e).__name__}: {str(e)[:400]}", flush=True)


run(21, 4096, 6144, "K=6144 baseline")
run(21, 4096, 704,  "K=704 production")
run(21, 4096, 5440, "K=5440 control")
run(21, 4096, 5472, "K=5472 production failure")
print("\nDONE", flush=True)
