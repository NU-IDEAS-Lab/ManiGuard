from maniguard.data.datagen.executor.variation import VariationSampler


class _Cand:
    """Minimal GraspCand stub — _params uses only .id, variants_stream only .reachable."""
    def __init__(self, cid, reachable=True):
        self.id = cid
        self.reachable = reachable


def _pull(sampler, cands, start_k, n):
    it = sampler.variants_stream(cands, start_k=start_k)
    return [next(it) for _ in range(n)]


def test_variants_stream_start_k_offsets_draws_disjoint():
    s = VariationSampler()
    cands = [_Cand(0), _Cand(1)]
    first = {p.seed for _, p in _pull(s, cands, 0, 20)}      # 2 grasps × k 0..9
    resumed = {p.seed for _, p in _pull(s, cands, 10, 20)}   # k 10..19
    assert first.isdisjoint(resumed)


def test_variants_stream_sets_draw_index():
    s = VariationSampler()
    cands = [_Cand(0), _Cand(1)]
    (c0, p0), (c1, p1), (c2, p2) = _pull(s, cands, 5, 3)
    assert p0.draw_index == 5 and p1.draw_index == 5   # both grasps at k=5
    assert p2.draw_index == 6                            # then k advances


def test_seed_encoding_collision_free_including_high_k():
    """grasp_id*1000+k aliased once k>=1000 (grasp0@k=1000 == grasp1@k=0). SeedSequence must not."""
    s = VariationSampler()
    seen = {}
    for gid in range(4):
        for k in range(0, 1200):
            vseed = s._params(_Cand(gid), k).seed
            assert vseed not in seen, f"collision: (gid={gid},k={k}) vs {seen.get(vseed)}"
            seen[vseed] = (gid, k)


def test_draw_zero_is_canonical():
    s = VariationSampler()
    p = s._params(_Cand(3), 0)
    assert p.jitter["above_xy"] == (0.0, 0.0)
    assert p.lift_clearance_mult == 1.0
    assert p.standoff_m == s.base_standoff
