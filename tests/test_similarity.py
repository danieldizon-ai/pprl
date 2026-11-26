import numpy as np
import numpy.typing as npt
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pprl.encoding import pack_bits
from pprl.similarity import dice_pairs, hamming_pairs, jaccard_pairs, popcount, qgram_dice

bitrows = arrays(np.bool_, st.tuples(st.integers(1, 6), st.just(130)))


@given(bitrows, bitrows)
def test_vectorised_metrics_match_reference(
    x: npt.NDArray[np.bool_], y: npt.NDArray[np.bool_]
) -> None:
    a, b = pack_bits(x), pack_bits(y)
    ia = np.repeat(np.arange(x.shape[0]), y.shape[0]).astype(np.int64)
    ib = np.tile(np.arange(y.shape[0]), x.shape[0]).astype(np.int64)
    d = dice_pairs(a, b, ia, ib)
    j = jaccard_pairs(a, b, ia, ib)
    h = hamming_pairs(a, b, ia, ib)
    for k, (i, jj) in enumerate(zip(ia, ib, strict=True)):
        inter = int((x[i] & y[jj]).sum())
        tot = int(x[i].sum() + y[jj].sum())
        union = int((x[i] | y[jj]).sum())
        assert d[k] == (2 * inter / tot if tot else 0.0)
        assert j[k] == (inter / union if union else 0.0)
        assert h[k] == int((x[i] ^ y[jj]).sum())
    nz = j > 0
    assert np.allclose(d[nz], 2 * j[nz] / (1 + j[nz]))


def test_popcount() -> None:
    bits = np.zeros((2, 70), dtype=bool)
    bits[0, [0, 63, 64, 69]] = True
    assert popcount(pack_bits(bits)).tolist() == [4, 0]


def test_empty_pairs_and_empty_filters() -> None:
    z = pack_bits(np.zeros((1, 64), dtype=bool))
    e = np.zeros(0, dtype=np.int64)
    assert dice_pairs(z, z, e, e).size == 0
    zero = np.zeros(1, dtype=np.int64)
    assert dice_pairs(z, z, zero, zero)[0] == 0.0


def test_qgram_dice() -> None:
    assert qgram_dice({"a", "b"}, {"b", "c"}) == 0.5
    assert qgram_dice(set(), set()) == 0.0
