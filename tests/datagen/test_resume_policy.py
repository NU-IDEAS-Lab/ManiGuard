from maniguard.data.datagen.executor.resume import compute_next_draw, resolve_start_k
from maniguard.data.datagen.executor.variation import VariationSampler


class _Cand:
    def __init__(self, cid, reachable=True):
        self.id = cid
        self.reachable = reachable


def test_resolve_start_k_precedence():
    assert resolve_start_k(None, None) == 0                    # fresh task
    assert resolve_start_k({}, None) == 0                      # pre-fix summary, no field
    assert resolve_start_k({"next_draw": 42}, None) == 42      # resume/top-up
    assert resolve_start_k({"next_draw": 42}, 100) == 100      # --start-draw override wins
    assert resolve_start_k(None, 0) == 0                       # override of 0 is honoured


def test_compute_next_draw():
    assert compute_next_draw(None, 7) == 7      # nothing ran → cursor unchanged
    assert compute_next_draw(9, 0) == 10        # ran up to k=9 → resume at 10


def test_resolve_start_k_ondisk_floor_beats_lost_summary():
    # summary's next_draw lost/stale (dedup wiped it, a crash skipped the write) but trajs on disk used
    # up to draw 18 → must resume at 19, never re-draw an existing k.
    assert resolve_start_k(None, None, ondisk_max_draw=18) == 19
    assert resolve_start_k({"next_draw": 1}, None, ondisk_max_draw=18) == 19   # stale small next_draw
    assert resolve_start_k({}, None, ondisk_max_draw=6) == 7                   # no next_draw key


def test_resolve_start_k_summary_beats_lower_ondisk():
    # summary's next_draw is higher than the on-disk max (it also skips FAILED k) → keep it
    assert resolve_start_k({"next_draw": 40}, None, ondisk_max_draw=18) == 40


def test_resolve_start_k_no_ondisk_trajs_unchanged():
    # default ondisk_max_draw=-1 → floor is 0 → original precedence preserved
    assert resolve_start_k(None, None) == 0
    assert resolve_start_k({"next_draw": 5}, None) == 5
    assert resolve_start_k({"next_draw": 5}, 3) == 3


def test_stop_resume_produces_no_seed_reuse():
    """Reproduces the task_0032 bug: pull 14 variants (7 draws × 2 grasps), stop, resume — the
    second round's seeds must be disjoint from the first."""
    s = VariationSampler()
    cands = [_Cand(0), _Cand(1)]
    it1 = s.variants_stream(cands, start_k=resolve_start_k(None, None))
    round1, last = [], None
    for _ in range(14):
        _, p = next(it1)
        round1.append(p.seed)
        last = p.draw_index
    nd = compute_next_draw(last, 0)
    it2 = s.variants_stream(cands, start_k=resolve_start_k({"next_draw": nd}, None))
    round2 = [next(it2)[1].seed for _ in range(14)]
    assert set(round1).isdisjoint(round2)
