import pytest

from ata.evaluate import calculate_metrics, time_ordered_split


def test_time_ordered_split_keeps_gap_and_never_shuffles():
    indices = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    train, test = time_ordered_split(indices, train_fraction=0.6, gap=2)
    assert train == indices[:6]
    assert test == indices[8:]
    assert max(train) < min(test)


def test_metrics_match_hand_computed_confusion_and_episode_delay():
    metrics = calculate_metrics(
        ["normal", "portscan", "normal", "ddos"],
        [False, True, True, False],
        window_indices=[0, 1, 2, 3],
        episodes=[
            {"type": "portscan", "start": 1.0, "end": 2.0},
            {"type": "ddos", "start": 3.0, "end": 4.0},
        ],
    )
    assert metrics["confusion"] == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["false_alarm_rate"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["per_attack_type_recall"] == {
        "ddos": pytest.approx(0.0),
        "portscan": pytest.approx(1.0),
    }
    assert [item["delay_seconds"] for item in metrics["detection_delay_per_episode_seconds"]] == [
        0.0,
        None,
    ]


def test_empty_metric_denominators_are_null_not_zero():
    metrics = calculate_metrics(["normal", "normal"], [False, False])
    assert metrics["precision"] is None
    assert metrics["recall"] is None
    assert metrics["false_alarm_rate"] == 0.0
    assert metrics["f1"] is None


@pytest.mark.parametrize(
    "indices,fraction,gap",
    [
        ([1, 1, 2], 0.5, 0),
        ([1, 2, 3], 0.5, -1),
        ([1, 2], 0.5, 1),
    ],
)
def test_invalid_ordered_splits_are_rejected(indices, fraction, gap):
    with pytest.raises(ValueError):
        time_ordered_split(indices, train_fraction=fraction, gap=gap)
