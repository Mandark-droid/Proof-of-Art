def test_demo_quilt_has_36_regions_and_only_replay_verified_patterns():
    from scripts.make_demo_gif import build_frames, build_state_quilt

    frame, coverage = build_state_quilt()

    assert frame.size == (1200, 700)
    assert coverage["state_count"] == 36
    assert coverage["verified_pattern_states"] == 29
    assert coverage["empty_states"] == 7
    assert len(build_frames()) == 6
