import json

import pytest

from ata.features import window_features
from ata.synth import build_attack_schedule, generate_synthetic_csv, generate_synthetic_data


def test_default_schedule_has_nine_episodes_and_eight_exact_gaps():
    episodes = build_attack_schedule(
        attack_start=120, n_cycles=3, attack_duration=30, attack_gap=40,
        seed=100,
    )
    assert len(episodes) == 9
    assert sum(
        episodes[index + 1]["start"] - episodes[index]["end"] == 40
        for index in range(len(episodes) - 1)
    ) == 8
    assert [episode["type"] for episode in episodes[:3]] == [
        "portscan", "ddos", "exfiltration"
    ]
    assert all(episode["duration"] == 30 for episode in episodes)
    assert episodes[-1]["end"] + 190 == 900


def test_attacks_make_at_least_25_windows_each_and_leave_normal_gaps_and_tail():
    flows = generate_synthetic_data(
        seed=101, warmup=5, attack_start=5, n_cycles=1,
        attack_duration=30, attack_gap=40, tail_duration=150,
    )
    windows = window_features(flows, window=1)
    counts = windows.loc[windows["label"].ne("normal"), "label"].value_counts()
    assert set(counts.index) == {"portscan", "ddos", "exfiltration"}
    assert all(int(counts[label]) >= 25 for label in counts.index)
    assert (flows.loc[flows["label"].ne("normal"), "ts"] >= 5).all()
    assert windows.loc[35:74, "label"].eq("normal").all()
    assert windows.loc[105:144, "label"].eq("normal").all()
    assert windows.loc[175:, "label"].eq("normal").all()
    assert len(windows) == 5 + 30 + 40 + 30 + 40 + 30 + 150


def test_default_total_duration_and_metadata_sidecar(tmp_path):
    output = tmp_path / "scenario.csv"
    flows = generate_synthetic_csv(output, seed=102)
    metadata_path = output.with_suffix(".csv.meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert len(metadata["episodes"]) == 9
    assert metadata["warmup_seconds"] == 120
    assert metadata["episodes"][0]["start"] == 120
    assert metadata["episodes"][-1]["end"] + 190 == 900
    assert metadata["duration_seconds"] == 900
    assert flows["ts"].max() < 900


def test_early_attacks_fail_and_shuffled_order_is_seeded():
    with pytest.raises(ValueError, match="earlier than the end of warm-up"):
        generate_synthetic_data(
            seed=100, warmup=10, attack_start=9, attacks=True, n_cycles=1,
            attack_duration=3, attack_gap=1, tail_duration=2,
        )
    with pytest.raises(ValueError, match="drift_start"):
        generate_synthetic_data(
            seed=100, warmup=10, seconds=40, attacks=False, drift=True,
            drift_start=9, drift_end=20,
        )
    first = build_attack_schedule(
        120, 6, 30, 40, attack_order="shuffled", seed=101
    )
    repeated = build_attack_schedule(
        120, 6, 30, 40, attack_order="shuffled", seed=101
    )
    changed = build_attack_schedule(
        120, 6, 30, 40, attack_order="shuffled", seed=102
    )
    assert first == repeated
    assert first != changed
    assert [item["type"] for item in first] != [
        item["type"] for item in changed
    ]
    for cycle in range(6):
        items = first[cycle * 3 : cycle * 3 + 3]
        assert {item["type"] for item in items} == {
            "portscan", "ddos", "exfiltration"
        }
        assert all(item["duration"] >= 25 for item in items)


def test_drift_is_independent_from_attacks_and_changes_late_rate():
    stable = generate_synthetic_data(
        seconds=60, seed=100, warmup=10, attacks=False
    )
    drifted = generate_synthetic_data(
        seconds=60, seed=100, warmup=10, attacks=False,
        drift=True, drift_start=10, drift_end=40,
    )
    assert set(drifted["label"].unique()) == {"normal"}
    early_rate = len(drifted.loc[drifted["ts"] < 10]) / 10
    late_rate = len(drifted.loc[drifted["ts"].between(40, 59.99)]) / 20
    stable_late_rate = len(stable.loc[stable["ts"].between(40, 59.99)]) / 20
    assert late_rate > early_rate
    assert late_rate > stable_late_rate

    no_attack_stealth = generate_synthetic_data(
        seconds=60,
        seed=102,
        warmup=10,
        attacks=False,
        stealth_start=10,
        stealth_duration=20,
        stealth_amp=2,
    )
    assert set(no_attack_stealth["label"].unique()) == {"normal"}


def test_inactive_stealth_is_not_reported_in_sidecar(tmp_path):
    output = tmp_path / "normal.csv"
    generate_synthetic_csv(
        output,
        seconds=60,
        seed=102,
        warmup=10,
        attacks=False,
        stealth_start=10,
        stealth_duration=20,
        stealth_amp=2,
    )
    metadata = json.loads(
        output.with_suffix(".csv.meta.json").read_text(encoding="utf-8")
    )
    assert metadata["stealth_start"] is None
