from ragblame.bench import FAULTS, _changes, summarize_bench
from ragblame.pipeline import Config
from ragblame.types import CAUSES


def case(true, reader, lp, rb, stages=()):
    return {"fault": "f", "true": true, "question_id": "q", "trace_reader": reader,
            "loss_point_only": lp, "ragblame": rb, "ragblame_stages": list(stages)}


def test_strict_scoring_excludes_multiple():
    s = summarize_bench([case("index", "index", "index", "multiple", ["index", "ranking"]),
                         case("ranking", "packing", "ranking", "ranking")])
    assert s["arms"] == {"trace_reader": 0.5, "loss_point_only": 1.0, "ragblame": 0.5}
    assert s["ragblame_multiple_including_true"] == 1


def test_per_stage_counts():
    s = summarize_bench([case("index", "index", "index", "index"), case("index", "x", "x", "index")])
    assert s["per_stage"]["index"] == {"trace_reader": "1/2", "loss_point_only": "1/2", "ragblame": "2/2"}


def test_every_fault_targets_a_known_cause_and_real_config_fields():
    fields = set(Config.__dataclass_fields__)
    for stage, spec in FAULTS.values():
        assert stage in CAUSES
        assert set(_changes(spec, frozenset({"d"}))) <= fields


def test_empty_bench_is_safe():
    assert summarize_bench([])["n"] == 0
