import os
import unittest
import torch
from sgl_kernel import minimax_h3_rotary_embedding


@unittest.skipUnless(os.environ.get("MINIMAX_H3_RUN_GPU_TESTS") == "1", "explicit free-device opt-in required")
class H3RotaryGpuTests(unittest.TestCase):
    def reference(self, value, positions, cache, head_size=128):
        half = cache.shape[1] // 2
        result = value.clone()
        view = result.view(value.shape[0], -1, head_size)
        source = value.view(value.shape[0], -1, head_size).double()
        a, b = source[..., :half], source[..., half:2 * half]
        c = cache[positions, :half].double().unsqueeze(1)
        s = cache[positions, half:].double().unsqueeze(1)
        if value.dtype == torch.bfloat16:
            y1 = (a*c).to(value.dtype).double() - (b*s).to(value.dtype).double()
            y2 = (b*c).to(value.dtype).double() + (a*s).to(value.dtype).double()
        else:
            y1 = (a*c - (b*s).to(value.dtype).double()).float()
            y2 = (b*c + (a*s).to(value.dtype).double()).float()
        view[..., :half] = y1.to(value.dtype)
        view[..., half:2 * half] = y2.to(value.dtype)
        return result

    def test_numerics_strided_qk_and_untouched_tail(self):
        generator = torch.Generator(device="cpu").manual_seed(17)
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            q_base = torch.randn(7, 3 * 128 + 11, generator=generator).to(dtype).cuda()
            k_base = torch.randn(7, 2 * 128 + 13, generator=generator).to(dtype).cuda()
            q, k = q_base[:, :3 * 128], k_base[:, :2 * 128]
            q0, k0 = q_base.cpu(), k_base.cpu()
            positions_buffer = (torch.arange(14) % 11).cuda()
            positions = positions_buffer[::2]
            cache = torch.randn(11, 96, generator=generator).to(dtype).cuda()
            expected_q = self.reference(q.cpu(), positions.cpu(), cache.cpu())
            expected_k = self.reference(k.cpu(), positions.cpu(), cache.cpu())
            minimax_h3_rotary_embedding(positions, q, k, 128, cache, True)
            torch.cuda.synchronize()
            torch.testing.assert_close(q.cpu(), expected_q, rtol=0, atol=0)
            torch.testing.assert_close(k.cpu(), expected_k, rtol=0, atol=0)
            self.assertTrue(torch.equal(q_base[:, 3 * 128:].cpu(), q0[:, 3 * 128:]))
            self.assertTrue(torch.equal(k_base[:, 2 * 128:].cpu(), k0[:, 2 * 128:]))

    def test_graph_replay_uses_changed_inputs_positions_and_cache(self):
        q = torch.ones(7, 3 * 128, device="cuda", dtype=torch.bfloat16)
        k = torch.ones(7, 2 * 128, device="cuda", dtype=torch.bfloat16)
        positions = torch.arange(7, device="cuda")
        cache = torch.ones(11, 96, device="cuda", dtype=torch.bfloat16)
        minimax_h3_rotary_embedding(positions, q, k, 128, cache, True)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            minimax_h3_rotary_embedding(positions, q, k, 128, cache, True)
        generator = torch.Generator(device="cpu").manual_seed(19)
        for step in (1, 2):
            q0 = torch.randn(q.shape, generator=generator).to(q.dtype)
            k0 = torch.randn(k.shape, generator=generator).to(k.dtype)
            p0 = (torch.arange(7) + step) % 11
            c0 = torch.randn(11, 96, generator=generator).to(cache.dtype)
            expected_q, expected_k = self.reference(q0, p0, c0), self.reference(k0, p0, c0)
            q.copy_(q0)
            k.copy_(k0)
            positions.copy_(p0)
            cache.copy_(c0)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(q.cpu(), expected_q, rtol=0, atol=0)
            torch.testing.assert_close(k.cpu(), expected_k, rtol=0, atol=0)
    def test_vae_graph_replay_uses_changed_inputs_positions_and_cache(self):
        q = torch.ones(7, 3 * 64, device="cuda", dtype=torch.bfloat16)
        k = torch.ones(7, 2 * 64, device="cuda", dtype=torch.bfloat16)
        positions = torch.arange(7, device="cuda")
        cache = torch.ones(11, 48, device="cuda", dtype=torch.bfloat16)
        minimax_h3_rotary_embedding(positions, q, k, 64, cache, True)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            minimax_h3_rotary_embedding(positions, q, k, 64, cache, True)
        generator = torch.Generator(device="cpu").manual_seed(19)
        for step in (1, 2):
            q0 = torch.randn(q.shape, generator=generator).to(q.dtype)
            k0 = torch.randn(k.shape, generator=generator).to(k.dtype)
            p0 = (torch.arange(7) + step) % 11
            c0 = torch.randn(11, 48, generator=generator).to(cache.dtype)
            expected_q, expected_k = self.reference(q0, p0, c0, head_size=64), self.reference(k0, p0, c0, head_size=64)
            q.copy_(q0)
            k.copy_(k0)
            positions.copy_(p0)
            cache.copy_(c0)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(q.cpu(), expected_q, rtol=0, atol=0)
            torch.testing.assert_close(k.cpu(), expected_k, rtol=0, atol=0)

if __name__ == "__main__":
    unittest.main()
