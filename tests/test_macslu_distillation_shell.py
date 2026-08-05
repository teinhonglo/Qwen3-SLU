import subprocess
import json
from pathlib import Path


def test_explicit_teacher_checkpoint_is_used_verbatim(tmp_path):
    checkpoint = tmp_path / "checkpoint-2820"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}")
    command = (
        "source local/macslu_distillation_lib.sh; "
        f"resolve_or_validate_teacher_checkpoint '{checkpoint}' ignored best"
    )
    result = subprocess.run(["bash", "-c", command], check=True, text=True,
                            capture_output=True)
    assert result.stdout.strip() == str(checkpoint.resolve())


def test_explicit_teacher_checkpoint_must_contain_model_files(tmp_path):
    checkpoint = tmp_path / "checkpoint-empty"
    checkpoint.mkdir()
    command = (
        "source local/macslu_distillation_lib.sh; "
        f"resolve_or_validate_teacher_checkpoint '{checkpoint}' ignored best"
    )
    result = subprocess.run(["bash", "-c", command], text=True,
                            capture_output=True)
    assert result.returncode != 0
    assert "neither a model/config nor a PEFT adapter" in result.stderr


def test_both_experiment_configs_keep_only_teacher_source_checkpoint():
    expected = "exp/macslu_fixed/macslu_qwen3_asr_17b_ep20_lora_woemblmhead/checkpoint-2820"
    configs = [
        "conf/macslu_qwen3_asr_06b_pruneslu_kd.json",
        "conf/macslu_qwen3_asr_06b_pruneslu_kd_kmeans.json",
    ]
    for config_path in configs:
        config = json.loads(Path(config_path).read_text())
        assert config[1]["teacher"]["teacher_source_checkpoint"] == expected
        assert "checkpoint_path" not in config[1]["teacher"]
        command = (
            "source local/macslu_distillation_lib.sh; "
            f"teacher_setting_from_conf '{config_path}' teacher_source_checkpoint"
        )
        result = subprocess.run(["bash", "-c", command], check=True, text=True,
                                capture_output=True)
        assert result.stdout.strip() == expected


def test_both_configs_share_vocabulary_and_teacher_source():
    configs = [
        json.loads(Path("conf/macslu_qwen3_asr_06b_pruneslu_kd.json").read_text()),
        json.loads(Path("conf/macslu_qwen3_asr_06b_pruneslu_kd_kmeans.json").read_text()),
    ]
    expected_source = "exp/macslu_fixed/macslu_qwen3_asr_17b_ep20_lora_woemblmhead/checkpoint-2820"
    assert configs[0][1]["teacher"]["teacher_source_checkpoint"] == expected_source
    assert configs[1][1]["teacher"]["teacher_source_checkpoint"] == expected_source
    assert configs[0][1]["vocabulary_pruning"] == configs[1][1]["vocabulary_pruning"]


def test_distillation_runner_contains_complete_pruneslu_flow():
    script = Path("run_macslu_distillation.sh").read_text()
    expected_stages = [
        "Prepare or verify MAC-SLU JSONL",
        "Compute MAC-SLU/Qwen3-ASR vocabulary",
        "Build or reuse vocabulary-pruned teacher with supervised fine-tuning",
        "Validate teacher/student compatibility",
        "Train student with teacher/student distillation",
        "Student inference",
        "MAC-SLU evaluation",
        "Summary",
    ]
    positions = [script.index(stage) for stage in expected_stages]
    assert positions == sorted(positions)


def test_vocabulary_pruning_is_optional_and_changes_exp_variant():
    for script_path in ("run_macslu_distillation.sh", "run_macslu_distillation_kmeans.sh"):
        script = Path(script_path).read_text()
        assert 'use_vocabulary_pruning=false' in script
        assert 'vocabulary_variant="fullvocab"' in script
        assert 'vocabulary_variant="vocabprune_top${vocabulary_top_frequency_tokens}"' in script
        assert '${student_tag}_${vocabulary_variant}' in script
        assert 'train_distillation_teacher' in script


def test_runtime_conf_toggles_vocabulary_pruning(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps([{}, {"teacher": {}}]))
    for enabled in ("true", "false"):
        output = tmp_path / f"{enabled}.json"
        subprocess.run([
            "python", "local/prepare_macslu_distillation_conf.py",
            "--input", str(source), "--output", str(output),
            "--vocabulary_pruning", enabled,
            "--vocabulary_manifest", "vocabulary.json",
            "--teacher_exp_dir", "teacher",
        ], check=True)
        config = json.loads(output.read_text())
        assert config[1]["vocabulary_pruning"]["enabled"] is (enabled == "true")
        assert config[1]["teacher"]["exp_dir"] == "teacher"
        assert "checkpoint_path" not in config[1]["teacher"]


def test_full_vocabulary_uses_teacher_source_and_stage2_only_runs_for_pruning():
    for script_path in ("run_macslu_distillation.sh", "run_macslu_distillation_kmeans.sh"):
        script = Path(script_path).read_text()
        assert 'conf_teacher_source=$(teacher_setting_from_conf "$base_student_train_conf" teacher_source_checkpoint)' in script
        assert 'teacher_for_student="$teacher_source_checkpoint"' in script
        assert 'if [ "$use_vocabulary_pruning" = true ]; then' in script
        assert 'Stage 2: Vocabulary pruning disabled; use teacher_source_checkpoint directly' in script
        assert 'teacher_checkpoint=""' not in script
        assert 'conf_teacher_checkpoint=' not in script
        assert '--teacher_checkpoint' not in script
