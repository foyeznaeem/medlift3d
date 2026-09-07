"""Gate G2: the projector is differentiable and `bp` is the exact adjoint of `fp`.

`test_projection_loss_has_gradient` is the assertion the FYDP-1 code needed and
did not have: its projector passed through DLPack with no autograd.Function, so
the projection loss was a constant and total variation was the entire objective.
"""
import numpy as np
import pytest
import torch


def test_adjoint_identity(any_projector):
    """<A x, y> == <x, A^T y> to float32 precision."""
    torch.manual_seed(0)
    P = any_projector
    x = torch.rand(P.grid.shape)
    y = torch.rand(P.proj_shape)
    lhs = float((P.fp(x) * y).sum())
    rhs = float((x * P.bp(y)).sum())
    assert lhs == pytest.approx(rhs, rel=1e-4)


def test_operator_is_linear(any_projector):
    torch.manual_seed(1)
    P = any_projector
    a, b = torch.rand(P.grid.shape), torch.rand(P.grid.shape)
    lhs = P.fp(a + 2.0 * b)
    rhs = P.fp(a) + 2.0 * P.fp(b)
    assert torch.allclose(lhs, rhs, atol=1e-4 * float(rhs.abs().max()))


def test_projection_loss_has_gradient(any_projector):
    """THE regression test for showstopper S1."""
    P = any_projector
    x = torch.rand(P.grid.shape, requires_grad=True)
    p = P.fp(x)
    assert p.requires_grad, "forward projection is detached from the graph"
    assert p.grad_fn is not None, "forward projection has no grad_fn"
    p.pow(2).sum().backward()
    assert x.grad is not None
    assert float(x.grad.abs().sum()) > 0


def test_bp_matches_autograd(any_projector):
    torch.manual_seed(2)
    P = any_projector
    y = torch.rand(P.proj_shape)
    x = torch.rand(P.grid.shape, requires_grad=True)
    (P.fp(x) * y).sum().backward()
    ref = P.bp(y)
    assert torch.allclose(x.grad, ref, atol=1e-4 * float(ref.abs().max()))


def test_bp_implementations_agree(any_projector):
    """The fast scatter path and the VJP path must be the same operator."""
    torch.manual_seed(3)
    y = torch.rand(any_projector.proj_shape)
    a = any_projector.bp_vjp(y)
    b = any_projector.bp_scatter(y)
    assert torch.allclose(a, b, atol=1e-4 * float(a.abs().max()))


def test_data_term_gradient_reduces_residual(any_projector):
    """Descending A^T(Ax-p) must actually reduce ||Ax-p||.

    If the gradient were fake -- constant, or coming only from a regulariser --
    this would not hold, which is precisely what went wrong in FYDP-1.
    """
    torch.manual_seed(4)
    P = any_projector
    truth = torch.rand(P.grid.shape)
    p = P.fp(truth)
    x = torch.zeros(P.grid.shape)
    r0 = float((P.fp(x) - p).pow(2).mean())
    for _ in range(8):
        g = P.bp(P.fp(x) - p)
        ag = P.fp(g)
        denom = float((ag * ag).sum())
        if denom < 1e-20:
            break
        # Exact steepest-descent step for the quadratic 0.5||Ax-p||^2.
        alpha = float((g * g).sum()) / denom
        x = x - alpha * g
    r1 = float((P.fp(x) - p).pow(2).mean())
    assert np.isfinite(r1), "descent diverged"
    assert r1 < 0.1 * r0, f"residual barely moved: {r0:.4g} -> {r1:.4g}"


def test_coverage_is_reasonable(parallel_projector):
    assert parallel_projector.coverage() > 0.3


def test_shape_validation(parallel_projector):
    with pytest.raises(ValueError):
        parallel_projector.fp(torch.zeros(3, 3, 3))
    with pytest.raises(ValueError):
        parallel_projector.bp(torch.zeros(3, 3))
