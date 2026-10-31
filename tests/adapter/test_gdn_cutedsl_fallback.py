"""CPU contracts for the HCU CuTeDSL GDN compatibility wrapper."""

import unittest

import torch

from hcu_adapter.gdn_cutedsl_fallback import _triton_gated_delta_rule_mtp


class GdnCutedslFallbackTests(unittest.TestCase):
    def test_adapts_layout_and_ring_contract(self):
        B, T, H, HV, K, V, slots = 2, 3, 2, 4, 5, 7, 6
        q = torch.randn(B, T, H, K)
        k = torch.randn_like(q)
        v = torch.randn(B, T, HV, V)
        a = torch.randn(B, T, HV)
        b = torch.randn_like(a)
        state = torch.randn(slots, HV, V, K)
        state_before = state.clone()
        indices = torch.tensor([1, 4], dtype=torch.int32)
        rings = {
            "replayssm_rawv": torch.empty(slots, HV, T, V),
            "replayssm_rawk": torch.empty(slots, H, T, K),
            "replayssm_g": torch.empty(slots, HV, T),
            "replayssm_beta": torch.empty(slots, HV, T),
        }
        seen = {}

        def recurrent(**kwargs):
            seen.update(kwargs)
            kwargs["initial_state_source"].add_(1)
            return torch.arange(B * T * HV * V).reshape(1, B * T, HV, V)

        def original(*args, **kwargs):
            raise AssertionError("CuTeDSL fallback unexpectedly declined")

        out = _triton_gated_delta_rule_mtp(
            original,
            recurrent,
            torch.randn(HV),
            a,
            torch.randn(HV),
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=state,
            initial_state_indices=indices,
            cache_ring=True,
            **rings,
        )
        self.assertEqual(out.shape, (B, T, HV, V))
        self.assertEqual(seen["q"].shape, (1, B * T, H, K))
        self.assertEqual(seen["initial_state_source"].shape, (slots, HV, K, V))
        self.assertTrue(
            torch.equal(
                seen["cu_seqlens"], torch.tensor([0, T, 2 * T], dtype=torch.int32)
            )
        )
        self.assertIs(seen["replayssm_rawv"], rings["replayssm_rawv"])
        self.assertTrue(torch.equal(state, state_before + 1))

    def test_declines_unknown_shape(self):
        marker = object()

        def original(*args, **kwargs):
            return marker

        result = _triton_gated_delta_rule_mtp(
            original,
            lambda **kwargs: None,
            torch.randn(2),
            torch.randn(2, 2, 2),
            torch.randn(2),
            q=torch.randn(2, 2, 2),
            k=torch.randn(2, 2, 2),
            v=torch.randn(2, 2, 2),
            b=torch.randn(2, 2),
            initial_state_source=torch.randn(2, 2, 2, 2),
            initial_state_indices=torch.tensor([0, 1]),
        )
        self.assertIs(result, marker)


if __name__ == "__main__":
    unittest.main()
