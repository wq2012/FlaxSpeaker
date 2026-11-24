#!/usr/bin/env python3
"""Generate every number and table of the FlaxSpeaker v0.7.0 paper from JSON archives.

Truth-in-reporting: the paper (`~/Code/github/flax_speaker_paper`) `\\input`s the files
written here, so no benchmark number is typed by hand. Sources:

- `pretrained_models/benchmark_results.json`           (55 converged models + 24 TFLite exports)
- `pretrained_models/full_data_flagship_results.json`  (11 Tier-2 full-data flagships + fusions)
- `pretrained_models/v070_ablation_results.json`       (7-arm 33k TPU v6p ablation + 4 horizon runs)
- `pretrained_models/eres2conformer_ablation_results.json` (3-arm 27.5k TPU v6p ablation)
- `pretrained_models/pfas_hparam_sweep.json`           (16-config PFAS hyperparameter sweep)
- `pretrained_models/ablation_results.json`            (7 converged Tier-1 ablation groups)
- `pretrained_models/vox1_o_10spk_subset_results.json` (VoxCeleb1-O 10-speaker subset, 55 models)
- `pretrained_models/r2_seed_and_control_results.json` (3-seed stability, horizon, matched data)
- `pretrained_models/phase1_convergence_pilot.json`    (5,000-step Phase-1 convergence pilot)
- `pretrained_models/step0_fingerprints.json`          (step-0 loss fingerprints, 55 checkpoints)

Usage: python scripts/generate_paper_tables.py --paper_dir ~/Code/github/flax_speaker_paper
"""

import argparse
import json
import os
import statistics
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PM = os.path.join(REPO_ROOT, "pretrained_models")

HIGHLIGHT_SPEC_ORDER = [
    "ls_conformer_small_arcface",
    "ls_conformer_small_ge2e",
    "cn_conformer_small_ge2e",
    "vox_conformer_small_ge2e",
]
HIGHLIGHT_UNIV_ORDER = [
    "univ_conformer_small_arcface",
    "univ_conformer_small_sphereface",
    "univ_conformer_small_cosface",
    "univ_conformer_small_ge2e_contrast",
    "univ_conformer_small_pfas_ge2e",
    "univ_lstm_small_ge2e",
    "univ_conformer_small_pfas",
    "univ_resnet_small_ge2e",
    "univ_conformer_small_ext_softmax",
    "univ_conformer_small_ge2e",
    "univ_conformer_small_dr_vectors",
    "univ_conformer_tiny_ge2e",
    "univ_conformer_base_ge2e",
    "univ_ecapa_tdnn_small_ge2e",
    "univ_transformer_small_ge2e",
    "univ_mamba_small_ge2e",
]
ALL_FLAGSHIP_ORDER = [
    "univ_full_conformer_large_dr_vectors",
    "univ_full_conformer_large_pfas",
    "univ_full_ecapa_c1024_arcface",
    "univ_full_conformer_large_arcface",
    "univ_full_resnet34_arcface",
    "univ_full_eres2conformer_large",
    "univ_full_eres2conformer_large_pfas",
    "univ_full_eres2conformer_v2_large",
    "univ_full_eres2conformer_v2_pfas",
    "univ_full_eres2conformer_xl_pfas",
    "univ_full_eres2conformer_xl",
    "univ_full_eres2conformer_v2_diar",
    "univ_full_eres2conformer_xl_diar",
    "univ_full_eres2conformer_xl_diar_pfas",
    "univ_full_eres2conformer_xl_diar_adapt",
]
TFLITE_REPOS = {
    "univ_full_conformer_large_arcface",
    "univ_full_ecapa_c1024_arcface",
    "univ_full_resnet34_arcface",
    "univ_full_eres2conformer_large",
    "univ_full_eres2conformer_v2_large",
    "univ_full_eres2conformer_xl",
    "univ_full_eres2conformer_v2_diar",
    "univ_full_eres2conformer_xl_diar",
    "univ_full_eres2conformer_xl_diar_adapt",
    "univ_conformer_small_ge2e",
    "univ_lstm_small_ge2e",
    "univ_ecapa_tdnn_small_ge2e",
    "ls_conformer_small_ge2e",
    "ls_lstm_small_ge2e",
    "ls_ecapa_tdnn_small_ge2e",
}
LOSS_SHORT = {
    "ge2e_softmax": "GE2E",
    "ge2e_contrast": "GE2E-C$^{*}$",
    "extended_set_softmax": "BXS$^{*}$",
    "arcface": "ArcFace",
    "subcenter_arcface": "SC-ArcFace",
    "cosface": "CosFace",
    "sphereface": "SphereFace",
    "triplet": "Triplet",
}
SCORE_SHORT = {
    "cosine": "cos",
    "dr_vector": "DrV$^{*}$",
    "pfas": "PFAS$^{*}$",
    "hybrid_pfas": "HybPFAS$^{*}$",
}


def tex_escape(s: str) -> str:
    return s.replace("_", "\\_")


def f2(x: float) -> str:
    return f"{x:.2f}"


def f4(x: float) -> str:
    return f"{x:.4f}"


def fmt_sci(p: float) -> str:
    if p < 1e-3:
        mant, expo = f"{p:.1e}".split("e")
        return f"{mant}\\times10^{{{int(expo)}}}"
    return f"{p:.2f}"


def load(name: str) -> Any:
    with open(os.path.join(PM, name), "r", encoding="utf-8") as f:
        return json.load(f)


def d1(e: dict, dom: str) -> dict:
    return e["multi_domain_eval"]["domains_1utt"][dom]


def d3(e: dict, dom: str) -> dict:
    return e["multi_domain_eval"]["domains_3utt"][dom]


def macro(e: dict, n: int) -> dict:
    raw_m = dict(e["multi_domain_eval"][f"universal_macro_avg_{n}utt"])
    dom_map = e["multi_domain_eval"][f"domains_{n}utt"]
    dom_keys = ("librispeech", "cnceleb2", "voxceleb1")
    for k_stat in ("min_dcf_001", "min_dcf_01", "roc_auc"):
        if k_stat not in raw_m and all(k_stat in dom_map.get(d, {}) for d in dom_keys):
            raw_m[k_stat] = sum(float(dom_map[d][k_stat]) for d in dom_keys) / 3.0
    return raw_m


def bold_max(values: list[float], fmt=f2) -> list[str]:
    best = max(values)
    return [f"\\textbf{{{fmt(v)}}}" if round(v, 6) == round(best, 6) else fmt(v) for v in values]


def bold_min(values: list[float], fmt=f2) -> list[str]:
    m = min(values)
    return [f"\\textbf{{{fmt(v)}}}" if round(v, 6) == round(m, 6) else fmt(v) for v in values]


def emit(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("% AUTO-GENERATED by FlaxSpeaker/scripts/generate_paper_tables.py -- do not edit.\n")
        f.write(text)


def macro_def(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def add_mde_nums(nums: list[str], prefix: str, mde: dict) -> None:
    p1 = mde["domains_1utt"]["universal_pooled"]
    p3 = mde["domains_3utt"]["universal_pooled"]
    nums.append(macro_def(prefix + "LsOne", f2(mde["domains_1utt"]["librispeech"]["eer_percent"])))
    nums.append(macro_def(prefix + "CnOne", f2(mde["domains_1utt"]["cnceleb2"]["eer_percent"])))
    nums.append(macro_def(prefix + "VoxOne", f2(mde["domains_1utt"]["voxceleb1"]["eer_percent"])))
    nums.append(macro_def(prefix + "MacroOne", f2(mde["universal_macro_avg_1utt"]["eer_percent"])))
    nums.append(macro_def(prefix + "MacroThree", f2(mde["universal_macro_avg_3utt"]["eer_percent"])))
    nums.append(macro_def(prefix + "PooledOne", f2(p1["eer_percent"])))
    nums.append(macro_def(prefix + "PooledOneLo", f2(p1["eer_ci95_low_percent"])))
    nums.append(macro_def(prefix + "PooledOneHi", f2(p1["eer_ci95_high_percent"])))
    nums.append(macro_def(prefix + "PooledThree", f2(p3["eer_percent"])))


def main(paper_dir: str) -> None:
    out_dir = os.path.join(paper_dir, "generated")
    os.makedirs(out_dir, exist_ok=True)
    bench = load("benchmark_results.json")
    abl = load("ablation_results.json")
    vox = load("vox1_o_10spk_subset_results.json")
    r2 = load("r2_seed_and_control_results.json")
    p1_pilot = load("phase1_convergence_pilot.json")
    fps = load("step0_fingerprints.json")
    flagship = load("full_data_flagship_results.json")
    pfas_sweep = load("pfas_hparam_sweep.json")

    ex = bench["experiments"]
    assert len(ex) in (44, 48, 49, 50, 55, 58, 59), len(ex)
    controlled_ex = {k: v for k, v in ex.items() if not k.startswith("univ_full_")}
    assert len(controlled_ex) == 44, len(controlled_ex)
    FLAGSHIP_ORDER = [m for m in ALL_FLAGSHIP_ORDER if m in ex]

    univ_order = sorted(
        [k for k in controlled_ex if k.startswith("univ_")],
        key=lambda k: (d1(ex[k], "universal_pooled")["eer_percent"], macro(ex[k], 1)["eer_percent"]),
    )
    single_order = sorted(
        [k for k in controlled_ex if not k.startswith("univ_")],
        key=lambda k: (
            {"librispeech": 0, "cnceleb2": 1, "voxceleb1": 2}[ex[k]["train_dataset"]],
            d1(ex[k], ex[k]["train_dataset"])["eer_percent"],
        ),
    )

    # ------------------------------------------------------------------ numbers.tex
    nums: list[str] = []
    times = sorted(float(e["train_time_sec"]) / 60.0 for e in controlled_ex.values())
    total_hours = sum(float(e["train_time_sec"]) for e in controlled_ex.values()) / 3600.0
    flagship_wall_hours = sum(
        float(e["train_time_sec"]) for k, e in ex.items() if k.startswith("univ_full_")
    ) / 3600.0
    flagship_tpu_hours = sum(
        float(e.get("num_devices", 4)) * float(e["train_time_sec"])
        for k, e in ex.items()
        if k.startswith("univ_full_")
    ) / 3600.0
    all_hours = sum(float(e["train_time_sec"]) for e in ex.values()) / 3600.0
    tmin = min(controlled_ex.items(), key=lambda kv: kv[1]["train_time_sec"])
    tmax = max(controlled_ex.items(), key=lambda kv: kv[1]["train_time_sec"])
    nums.append(macro_def("numModels", str(len(controlled_ex))))
    nums.append(macro_def("numUnivTierOne", str(len(univ_order))))
    nums.append(macro_def("numSingleTierOne", str(len(single_order))))
    nums.append(macro_def("numTotalModels", str(len(ex))))
    nums.append(macro_def("numFlagshipModels", str(len(FLAGSHIP_ORDER))))
    nums.append(macro_def("trainMinMinutes", f"{times[0]:.1f}"))
    nums.append(macro_def("trainMaxMinutes", f"{times[-1]:.1f}"))
    nums.append(macro_def("trainMedianMinutes", f"{statistics.median(times):.1f}"))
    nums.append(macro_def("trainTotalHours", f"{total_hours:.2f}"))
    nums.append(macro_def("trainFlagshipWallHours", f"{flagship_wall_hours:.2f}"))
    nums.append(macro_def("trainFlagshipTpuHours", f"{flagship_tpu_hours:.2f}"))
    nums.append(macro_def("trainAllModelsHours", f"{all_hours:.2f}"))
    nums.append(macro_def("trainMinModel", tex_escape(tmin[0])))
    nums.append(macro_def("trainMaxModel", tex_escape(tmax[0])))

    params = [int(e["num_parameters"]) for e in controlled_ex.values()]
    all_params = [int(e["num_parameters"]) for e in ex.values()]
    nums.append(macro_def("paramsMinM", f"{min(params) / 1e6:.2f}"))
    nums.append(macro_def("paramsMaxM", f"{max(params) / 1e6:.2f}"))
    nums.append(macro_def("allParamsMaxM", f"{max(all_params) / 1e6:.2f}"))

    pooled_all = [d1(e, "universal_pooled")["eer_percent"] for e in controlled_ex.values()]
    pooled_univ = [d1(ex[k], "universal_pooled")["eer_percent"] for k in univ_order]
    pooled3_all = [d3(e, "universal_pooled")["eer_percent"] for e in controlled_ex.values()]
    nums.append(macro_def("zooPooledOneMin", f2(min(pooled_all))))
    nums.append(macro_def("zooPooledOneMax", f2(max(pooled_all))))
    nums.append(macro_def("zooUnivPooledOneMin", f2(min(pooled_univ))))
    nums.append(macro_def("zooUnivPooledOneMax", f2(max(pooled_univ))))
    nums.append(macro_def("zooPooledThreeMin", f2(min(pooled3_all))))
    nums.append(macro_def("zooPooledThreeMax", f2(max(pooled3_all))))
    losses_used = sorted({e["loss_type"] for e in controlled_ex.values()})
    nums.append(macro_def("numLossesBenchmarked", str(len(losses_used))))

    def add_model_nums(prefix: str, mid: str) -> None:
        e = ex[mid]
        add_mde_nums(nums, prefix, e["multi_domain_eval"])
        nums.append(macro_def(prefix + "LsThree", f2(d3(e, "librispeech")["eer_percent"])))
        nums.append(macro_def(prefix + "CnThree", f2(d3(e, "cnceleb2")["eer_percent"])))
        nums.append(macro_def(prefix + "VoxThree", f2(d3(e, "voxceleb1")["eer_percent"])))
        nums.append(macro_def(prefix + "MacroMinDcfOne", f4(macro(e, 1)["min_dcf_001"])))
        nums.append(macro_def(prefix + "MacroMinDcfThree", f4(macro(e, 3)["min_dcf_001"])))
        nums.append(macro_def(prefix + "PooledAucOne", f4(d1(e, "universal_pooled")["roc_auc"])))
        nums.append(macro_def(prefix + "PooledAucThree", f4(d3(e, "universal_pooled")["roc_auc"])))
        nums.append(macro_def(prefix + "VoxSubset", f2(e["cross_dataset_eval"]["vox1_o_10spk_subset"]["eer_percent"])))
        nums.append(macro_def(prefix + "ParamsM", f"{int(e['num_parameters']) / 1e6:.2f}"))
        nums.append(macro_def(prefix + "Steps", f"{int(e['num_steps']):,}"))
        nums.append(macro_def(prefix + "InitLoss", f"{float(e['initial_loss']):.2f}"))
        nums.append(macro_def(prefix + "FinalLoss", f"{float(e['final_loss']):.2f}"))
        nums.append(macro_def(prefix + "InitLossFour", f"{float(e['initial_loss']):.4f}"))
        nums.append(macro_def(prefix + "FinalLossFour", f"{float(e['final_loss']):.4f}"))

    for prefix, mid in [
        ("univSmall", "univ_conformer_small_ge2e"),
        ("univTiny", "univ_conformer_tiny_ge2e"),
        ("univBase", "univ_conformer_base_ge2e"),
        ("lsSmall", "ls_conformer_small_ge2e"),
        ("lsTiny", "ls_conformer_tiny_ge2e"),
        ("lsBase", "ls_conformer_base_ge2e"),
        ("lsArc", "ls_conformer_small_arcface"),
        ("lsCos", "ls_conformer_small_cosface"),
        ("lsSphere", "ls_conformer_small_sphereface"),
        ("cnSmall", "cn_conformer_small_ge2e"),
        ("cnEcapaArc", "cn_ecapa_tdnn_small_arcface"),
        ("voxSmall", "vox_conformer_small_ge2e"),
        ("univArc", "univ_conformer_small_arcface"),
        ("univCos", "univ_conformer_small_cosface"),
        ("univSphere", "univ_conformer_small_sphereface"),
        ("univContrast", "univ_conformer_small_ge2e_contrast"),
        ("univExt", "univ_conformer_small_ext_softmax"),
        ("univDr", "univ_conformer_small_dr_vectors"),
        ("univPfas", "univ_conformer_small_pfas"),
        ("univPfasGe", "univ_conformer_small_pfas_ge2e"),
        ("univTriplet", "univ_conformer_small_triplet"),
        ("univMamba", "univ_mamba_small_ge2e"),
        ("univTransformer", "univ_transformer_small_ge2e"),
        ("univLstm", "univ_lstm_small_ge2e"),
        ("univEcapaT", "univ_ecapa_tdnn_tiny_ge2e"),
        ("univEcapaS", "univ_ecapa_tdnn_small_ge2e"),
        ("univEcapaB", "univ_ecapa_tdnn_base_ge2e"),
        ("lsEcapaT", "ls_ecapa_tdnn_tiny_ge2e"),
        ("lsEcapaS", "ls_ecapa_tdnn_small_ge2e"),
        ("lsEcapaB", "ls_ecapa_tdnn_base_ge2e"),
        ("univResnet", "univ_resnet_small_ge2e"),
        ("lsResnet", "ls_resnet_small_ge2e"),
        ("cnDr", "cn_conformer_small_dr_vectors"),
        ("voxEcapaXs", "vox_ecapa_tdnn_small_ext_softmax"),
        ("flagConf", "univ_full_conformer_large_arcface"),
        ("flagEcapa", "univ_full_ecapa_c1024_arcface"),
        ("flagResnet", "univ_full_resnet34_arcface"),
        ("flagEres", "univ_full_eres2conformer_large"),
        ("flagEresPfas", "univ_full_eres2conformer_large_pfas"),
        ("flagVtwo", "univ_full_eres2conformer_v2_large"),
        ("flagVtwoPfas", "univ_full_eres2conformer_v2_pfas"),
        ("flagXl", "univ_full_eres2conformer_xl"),
        ("flagXlPfas", "univ_full_eres2conformer_xl_pfas"),
        ("flagVtwoDiar", "univ_full_eres2conformer_v2_diar"),
        ("flagXlDiar", "univ_full_eres2conformer_xl_diar"),
        ("flagXlDiarPfas", "univ_full_eres2conformer_xl_diar_pfas"),
        ("flagXlAdapt", "univ_full_eres2conformer_xl_diar_adapt"),
        ("flagPfas", "univ_full_conformer_large_pfas"),
        ("flagDr", "univ_full_conformer_large_dr_vectors"),
    ]:
        if mid in ex:
            add_model_nums(prefix, mid)

    # Tier-2 Flagship & Score Fusion macros.
    sm = flagship["shard_metadata"]
    nums.append(macro_def("flagshipSpks", f"{sm['num_base_speakers']:,}"))
    nums.append(macro_def("flagshipClasses", f"{sm['num_speakers']:,}"))
    nums.append(macro_def("flagshipUtts", f"{sm['clean_utterances']:,}"))
    nums.append(macro_def("flagshipItems", f"{sm['num_items']:,}"))
    nums.append(macro_def("flagshipShards", str(sm["num_shards"])))

    sm7 = flagship.get("v080_7corpus_shard_metadata", {})
    if sm7:
        nums.append(macro_def("sevenCorpSpks", f"{sm7['num_base_speakers']:,}"))
        nums.append(macro_def("sevenCorpClasses", f"{sm7['num_speakers']:,}"))
        nums.append(macro_def("sevenCorpUtts", f"{sm7['clean_utterances']:,}"))
        nums.append(macro_def("sevenCorpItems", f"{sm7['num_items']:,}"))
        nums.append(macro_def("sevenCorpShards", str(sm7["num_shards"])))
        nums.append(macro_def("diarSpks", f"{sm7['diarization_base_speakers']:,}"))
        nums.append(macro_def("diarClasses", f"{sm7['diarization_speed_perturbed_classes']:,}"))
        nums.append(macro_def("diarUtts", f"{sm7['diarization_clean_turns']:,}"))
        nums.append(macro_def("diarItems", f"{sm7['diarization_augmented_items']:,}"))
        nums.append(macro_def("diarShards", str(sm7["diarization_shards"])))

    fm = flagship["models"]
    for ftag, fmid in (
        ("flagConf", "univ_full_conformer_large_arcface"),
        ("flagEcapa", "univ_full_ecapa_c1024_arcface"),
        ("flagResnet", "univ_full_resnet34_arcface"),
        ("flagEres", "univ_full_eres2conformer_large"),
        ("flagEresPfas", "univ_full_eres2conformer_large_pfas"),
        ("flagVtwo", "univ_full_eres2conformer_v2_large"),
        ("flagVtwoPfas", "univ_full_eres2conformer_v2_pfas"),
        ("flagXl", "univ_full_eres2conformer_xl"),
        ("flagXlPfas", "univ_full_eres2conformer_xl_pfas"),
        ("flagVtwoDiar", "univ_full_eres2conformer_v2_diar"),
        ("flagXlDiar", "univ_full_eres2conformer_xl_diar"),
        ("flagXlDiarPfas", "univ_full_eres2conformer_xl_diar_pfas"),
        ("flagXlAdapt", "univ_full_eres2conformer_xl_diar_adapt"),
        ("flagPfas", "univ_full_conformer_large_pfas"),
        ("flagDr", "univ_full_conformer_large_dr_vectors"),
    ):
        if fmid not in fm:
            continue
        ob = fm[fmid]["official_benchmarks"]
        vx, cn, ls = ob["voxceleb1_o"], ob["cnceleb_e"], ob["librispeech_test_clean"]
        nums.append(macro_def(ftag + "VoxRaw", f2(vx["eer_percent"])))
        if "eer_ci95_low_percent" in vx:
            nums.append(macro_def(ftag + "VoxRawLo", f2(vx["eer_ci95_low_percent"])))
            nums.append(macro_def(ftag + "VoxRawHi", f2(vx["eer_ci95_high_percent"])))
        nums.append(macro_def(ftag + "VoxQmf", f2(vx["qmf"]["eer_percent"])))
        if "eer_ci95_low_percent" in vx["qmf"]:
            nums.append(macro_def(ftag + "VoxQmfLo", f2(vx["qmf"]["eer_ci95_low_percent"])))
            nums.append(macro_def(ftag + "VoxQmfHi", f2(vx["qmf"]["eer_ci95_high_percent"])))
        nums.append(macro_def(ftag + "VoxDcf", f4(vx["min_dcf_001"])))
        nums.append(macro_def(ftag + "VoxDcfQmf", f4(vx["qmf"]["min_dcf_001"])))
        nums.append(macro_def(ftag + "VoxAuc", f4(vx["roc_auc"])))
        nums.append(macro_def(ftag + "CnRaw", f2(cn["eer_percent"])))
        nums.append(macro_def(ftag + "CnQmf", f2(cn["qmf"]["eer_percent"])))
        nums.append(macro_def(ftag + "CnDcf", f4(cn["min_dcf_001"])))
        nums.append(macro_def(ftag + "CnDcfQmf", f4(cn["qmf"]["min_dcf_001"])))
        cn_me_d = cn["multi_enroll_map"].get("as_norm", cn["multi_enroll_map"])
        cn_me = cn_me_d["eer_percent"]
        nums.append(macro_def(ftag + "CnMulti", f2(cn_me)))
        if "min_dcf_001" in cn_me_d:
            nums.append(macro_def(ftag + "CnMultiDcf", f4(cn_me_d["min_dcf_001"])))
        nums.append(macro_def(ftag + "LsRaw", f2(ls["eer_percent"])))
        nums.append(macro_def(ftag + "LsQmf", f2(ls["qmf"]["eer_percent"])))
        nums.append(
            macro_def(ftag + "LsMultiRaw", f2(ls["multi_enroll_3utt"]["eer_percent"]))
        )
        ls_me = ls["multi_enroll_3utt"].get(
            "as_norm", ls["multi_enroll_3utt"]
        )["eer_percent"]
        nums.append(macro_def(ftag + "LsMulti", f2(ls_me)))
        if "as_norm" in vx:
            nums.append(macro_def(ftag + "VoxAsn", f2(vx["as_norm"]["eer_percent"])))
            nums.append(macro_def(ftag + "VoxDcfAsn", f4(vx["as_norm"]["min_dcf_001"])))
            nums.append(macro_def(ftag + "CnAsn", f2(cn["as_norm"]["eer_percent"])))
            nums.append(macro_def(ftag + "LsAsn", f2(ls["as_norm"]["eer_percent"])))
        ds_st = fm[fmid].get("dataset_stats", {})
        if "total_utterances" in ds_st:
            nums.append(
                macro_def(ftag + "Utts", f"{int(ds_st['total_utterances']):,}")
            )
            nums.append(
                macro_def(ftag + "UttsM", f"{int(ds_st['total_utterances']) / 1e6:.2f}")
            )
            nums.append(
                macro_def(
                    ftag + "Items",
                    f"{int(ds_st['total_augmented_shard_items']):,}",
                )
            )
        ov_st = fm[fmid].get("step0_phase_overrides", {})
        if "max_total_shards" in ov_st:
            nums.append(
                macro_def(ftag + "Shards", str(int(ov_st["max_total_shards"])))
            )

    fus = flagship["heterogeneous_fusion"]
    fus2 = fus["official_benchmarks"]
    fus3 = fus["multi_enroll_3way_fusion"]
    if "two_way_official_benchmarks" in fus:
        fus_2w_m = fus["two_way_official_benchmarks"]
        nums.append(macro_def("fusionTwoWVoxAsn", f2(fus_2w_m["voxceleb1_o"]["eer_percent"])))
        nums.append(macro_def("fusionTwoWVoxQmf", f2(fus_2w_m["voxceleb1_o"]["qmf"]["eer_percent"])))
        nums.append(macro_def("fusionTwoWCnQmf", f2(fus_2w_m["cnceleb_e"]["qmf"]["eer_percent"])))
        nums.append(macro_def("fusionTwoWLsQmf", f2(fus_2w_m["librispeech_test_clean"]["qmf"]["eer_percent"])))
    if "three_way_backbone_official_benchmarks" in fus:
        fus_3bb_m = fus["three_way_backbone_official_benchmarks"]
        nums.append(macro_def("fusionThreeBbVoxAsn", f2(fus_3bb_m["voxceleb1_o"]["eer_percent"])))
        nums.append(macro_def("fusionThreeBbVoxQmf", f2(fus_3bb_m["voxceleb1_o"]["qmf"]["eer_percent"])))
        nums.append(macro_def("fusionThreeBbCnQmf", f2(fus_3bb_m["cnceleb_e"]["qmf"]["eer_percent"])))
        nums.append(macro_def("fusionThreeBbLsQmf", f2(fus_3bb_m["librispeech_test_clean"]["qmf"]["eer_percent"])))
    if "four_way_backbone_official_benchmarks" in fus:
        fus_4bb_m = fus["four_way_backbone_official_benchmarks"]
        nums.append(macro_def("fusionFourBbVoxAsn", f2(fus_4bb_m["voxceleb1_o"]["eer_percent"])))
        nums.append(macro_def("fusionFourBbVoxQmf", f2(fus_4bb_m["voxceleb1_o"]["qmf"]["eer_percent"])))
        nums.append(macro_def("fusionFourBbCnQmf", f2(fus_4bb_m["cnceleb_e"]["qmf"]["eer_percent"])))
        nums.append(macro_def("fusionFourBbLsQmf", f2(fus_4bb_m["librispeech_test_clean"]["qmf"]["eer_percent"])))
    if "six_way_cosine_official_benchmarks" in fus:
        fus_6c_m = fus["six_way_cosine_official_benchmarks"]
        nums.append(macro_def("fusionSixCosVoxAsn", f2(fus_6c_m["voxceleb1_o"]["eer_percent"])))
        nums.append(macro_def("fusionSixCosVoxQmf", f2(fus_6c_m["voxceleb1_o"]["qmf"]["eer_percent"])))
        nums.append(macro_def("fusionSixCosCnQmf", f2(fus_6c_m["cnceleb_e"]["qmf"]["eer_percent"])))
        nums.append(macro_def("fusionSixCosLsQmf", f2(fus_6c_m["librispeech_test_clean"]["qmf"]["eer_percent"])))
    nums.append(macro_def("fusionSixHybVoxAsn", f2(fus2["voxceleb1_o"]["eer_percent"])))
    nums.append(macro_def("fusionSixHybVoxQmf", f2(fus2["voxceleb1_o"]["qmf"]["eer_percent"])))
    nums.append(macro_def("fusionSixHybCnQmf", f2(fus2["cnceleb_e"]["qmf"]["eer_percent"])))
    nums.append(macro_def("fusionSixHybLsQmf", f2(fus2["librispeech_test_clean"]["qmf"]["eer_percent"])))
    nums.append(macro_def("fusionSixMeCnMulti", f2(fus3["cnceleb_e_multi_enroll_map"]["eer_percent"])))
    nums.append(macro_def("fusionSixMeCnMultiAuc", f4(fus3["cnceleb_e_multi_enroll_map"]["roc_auc"])))
    nums.append(macro_def("fusionSixMeLsMulti", f2(fus3["librispeech_test_clean_3utt"]["eer_percent"])))
    nums.append(macro_def("fusionSixMeLsMultiAuc", f4(fus3["librispeech_test_clean_3utt"]["roc_auc"])))
    if "v080_7corpus_official_benchmarks" in fus:
        fus_v8_m = fus["v080_7corpus_official_benchmarks"]
        nums.append(macro_def("fusionSevenCorpVoxAsn", f2(fus_v8_m["voxceleb1_o"]["eer_percent"])))
        nums.append(macro_def("fusionSevenCorpVoxQmf", f2(fus_v8_m["voxceleb1_o"]["qmf"]["eer_percent"])))
        nums.append(macro_def("fusionSevenCorpCnQmf", f2(fus_v8_m["cnceleb_e"]["qmf"]["eer_percent"])))
        nums.append(macro_def("fusionSevenCorpLsQmf", f2(fus_v8_m["librispeech_test_clean"]["qmf"]["eer_percent"])))
    if "v080_7corpus_multi_enroll_fusion" in fus:
        fus_v8_me = fus["v080_7corpus_multi_enroll_fusion"]
        nums.append(macro_def("fusionSevenCorpCnMulti", f2(fus_v8_me["cnceleb_e_multi_enroll_map"]["eer_percent"])))
        nums.append(macro_def("fusionSevenCorpLsMulti", f2(fus_v8_me["librispeech_test_clean_3utt"]["eer_percent"])))

    v080_abl_path = os.path.join(PM, "v080_diarization_ablation_results.json")
    v080_abl = None
    if os.path.isfile(v080_abl_path):
        with open(v080_abl_path, "r", encoding="utf-8") as f:
            v080_abl = json.load(f)

        def _add_diar_macros(dtag: str, db_d: dict) -> None:
            sd = db_d["suites"]
            md = db_d["macro_averages"]
            nums.append(macro_def(dtag + "FisherWsRaw", f2(sd["fisher_test_within_session"]["eer_percent"])))
            nums.append(macro_def(dtag + "FisherWsQmf", f2(sd["fisher_test_within_session"]["qmf"]["eer_percent"])))
            nums.append(macro_def(dtag + "FisherStdQmf", f2(sd["fisher_test_standard"]["qmf"]["eer_percent"])))
            nums.append(macro_def(dtag + "FisherShortQmf", f2(sd["fisher_test_short_turn"]["qmf"]["eer_percent"])))
            nums.append(macro_def(dtag + "CallhomeWsQmf", f2(sd["callhome_test_within_session"]["qmf"]["eer_percent"])))
            nums.append(macro_def(dtag + "CallhomeStdQmf", f2(sd["callhome_test_standard"]["qmf"]["eer_percent"])))
            nums.append(macro_def(dtag + "CallhomeShortQmf", f2(sd["callhome_test_short_turn"]["qmf"]["eer_percent"])))
            nums.append(macro_def(dtag + "AmiWsRaw", f2(sd["ami_test_within_session"]["eer_percent"])))
            nums.append(macro_def(dtag + "AmiWsQmf", f2(sd["ami_test_within_session"]["qmf"]["eer_percent"])))
            nums.append(macro_def(dtag + "AmiStdQmf", f2(sd["ami_test_standard"]["qmf"]["eer_percent"])))
            nums.append(macro_def(dtag + "AmiShortQmf", f2(sd["ami_test_short_turn"]["qmf"]["eer_percent"])))
            nums.append(macro_def(dtag + "IcsiWsQmf", f2(sd["icsi_test_within_session"]["qmf"]["eer_percent"])))
            nums.append(macro_def(dtag + "IcsiStdQmf", f2(sd["icsi_test_standard"]["qmf"]["eer_percent"])))
            nums.append(macro_def(dtag + "IcsiShortQmf", f2(sd["icsi_test_short_turn"]["qmf"]["eer_percent"])))
            nums.append(macro_def(dtag + "MacroWsRaw", f2(md["macro_4corpus_within_session"]["eer_percent"])))
            nums.append(macro_def(dtag + "MacroWsAsn", f2(md["macro_4corpus_within_session"]["as_norm"]["eer_percent"])))
            nums.append(macro_def(dtag + "MacroWsQmf", f2(md["macro_4corpus_within_session"]["qmf"]["eer_percent"])))
            nums.append(macro_def(dtag + "MacroStdRaw", f2(md["macro_4corpus_standard"]["eer_percent"])))
            nums.append(macro_def(dtag + "MacroStdQmf", f2(md["macro_4corpus_standard"]["qmf"]["eer_percent"])))
            nums.append(macro_def(dtag + "MacroShortRaw", f2(md["macro_4corpus_short_turn"]["eer_percent"])))
            nums.append(macro_def(dtag + "MacroShortQmf", f2(md["macro_4corpus_short_turn"]["qmf"]["eer_percent"])))

        zs_map = v080_abl.get("v070_zero_shot_models", {})
        for ztag, zmid in (
            ("zsEcapa", "univ_full_ecapa_c1024_arcface"),
            ("zsResnet", "univ_full_resnet34_arcface"),
            ("zsEres", "univ_full_eres2conformer_large"),
            ("zsVtwo", "univ_full_eres2conformer_v2_large"),
            ("zsVtwoPfas", "univ_full_eres2conformer_v2_pfas"),
            ("zsXl", "univ_full_eres2conformer_xl"),
            ("zsXlPfas", "univ_full_eres2conformer_xl_pfas"),
        ):
            if zmid in zs_map:
                _add_diar_macros(ztag, zs_map[zmid]["diarization_benchmarks"])
        rel_map = v080_abl.get("v080_released_flagships", {})
        for rtag, rmid in (
            ("flagVtwoDiar", "univ_full_eres2conformer_v2_diar"),
            ("flagXlDiar", "univ_full_eres2conformer_xl_diar"),
            ("flagXlDiarPfas", "univ_full_eres2conformer_xl_diar_pfas"),
            ("flagXlAdapt", "univ_full_eres2conformer_xl_diar_adapt"),
        ):
            if rmid in rel_map:
                _add_diar_macros(rtag, rel_map[rmid]["diarization_benchmarks"])
        fus_d_map = v080_abl.get("v080_7corpus_fusion", {})
        if "diarization_benchmarks" in fus_d_map:
            _add_diar_macros("fusionSevenCorp", fus_d_map["diarization_benchmarks"])

    v070_abl_path = os.path.join(PM, "v070_ablation_results.json")
    v070_abl = None
    if os.path.isfile(v070_abl_path):
        with open(v070_abl_path, "r", encoding="utf-8") as f:
            v070_abl = json.load(f)
        for vtag, vaid in (
            ("vZeroBase", "ctrl_base_noprune"),
            ("vZeroMedoid", "clean_lmft_medoid_k1"),
            ("vZeroKeepSwa", "clean_lmft_keep_k3_swa_bn"),
            ("vZeroInterShort", "inter_topk_chan_aug_k3"),
            ("vZeroVtwoShort", "v2_large_fwse_mfa123"),
            ("vZeroXlShort", "xl_64m_clean_k3"),
            ("vZeroHybShort", "hybrid_pfas_attentive_scoring"),
        ):
            if vaid in v070_abl.get("arms", {}):
                va_d = v070_abl["arms"][vaid]
                v_ob = va_d["official_benchmarks"]
                nums.append(macro_def(vtag + "VoxRaw", f2(v_ob["voxceleb1_o"]["eer_percent"])))
                nums.append(macro_def(vtag + "VoxQmf", f2(v_ob["voxceleb1_o"]["qmf"]["eer_percent"])))
                nums.append(macro_def(vtag + "CnMulti", f2(v_ob["cnceleb_e"]["multi_enroll_map"]["as_norm"]["eer_percent"])))
                if "final_loss" in va_d:
                    nums.append(macro_def(vtag + "Loss", f"{va_d['final_loss']:.2f}"))
                vt_arm = (va_d.get("validation_trajectory") or va_d.get("validation_history") or [{}])[-1]
                if "val_1utt_eer_percent" in vt_arm:
                    nums.append(macro_def(vtag + "ValOne", f2(vt_arm["val_1utt_eer_percent"])))
                if "val_composite_eer_percent" in vt_arm:
                    nums.append(macro_def(vtag + "ValComp", f2(vt_arm["val_composite_eer_percent"])))
        lh_runs = v070_abl.get("long_horizon_recipe_comparison", {}).get("runs", {})
        for ltag, lkey in (
            ("lhVtwoClean", "v2_large_clean_162k"),
            ("lhVtwoInter", "v2_large_interaug_162k"),
            ("lhXlClean", "xl_64m_clean_145k"),
            ("lhXlInter", "xl_64m_interaug_145k"),
        ):
            if lkey in lh_runs:
                lr_d = lh_runs[lkey]
                lob = lr_d["official_benchmarks"]
                vt_l = lr_d["validation_trajectory"][-1]
                nums.append(macro_def(ltag + "Val", f2(vt_l["val_1utt_eer_percent"])))
                nums.append(macro_def(ltag + "ValComp", f2(vt_l["val_composite_eer_percent"])))
                nums.append(macro_def(ltag + "Loss", f"{lr_d['final_loss']:.2f}"))
                nums.append(macro_def(ltag + "VoxRaw", f2(lob["voxceleb1_o"]["eer_percent"])))
                nums.append(macro_def(ltag + "VoxQmf", f2(lob["voxceleb1_o"]["qmf"]["eer_percent"])))
                nums.append(macro_def(ltag + "CnMulti", f2(lob["cnceleb_e"]["multi_enroll_map"]["as_norm"]["eer_percent"])))
                nums.append(macro_def(ltag + "LsQmf", f2(lob["librispeech_test_clean"]["qmf"]["eer_percent"])))

    eres_abl_path = os.path.join(PM, "eres2conformer_ablation_results.json")
    eres_abl = None
    if os.path.isfile(eres_abl_path):
        with open(eres_abl_path, "r", encoding="utf-8") as f:
            eres_abl = json.load(f)
        for atag, aid in (
            ("ablEresSgd", "eres2conformer_catp_sgd"),
            ("ablEresAdam", "eres2conformer_catp_adamw"),
            ("ablEresNoConf", "eres2net_noconf_catp_sgd"),
        ):
            arm_d = eres_abl["arms"][aid]
            a_ob = arm_d["official_benchmarks"]
            nums.append(macro_def(atag + "VoxRaw", f2(a_ob["voxceleb1_o"]["eer_percent"])))
            nums.append(macro_def(atag + "VoxAsn", f2(a_ob["voxceleb1_o"]["as_norm"]["eer_percent"])))
            nums.append(macro_def(atag + "VoxQmf", f2(a_ob["voxceleb1_o"]["qmf"]["eer_percent"])))
            nums.append(macro_def(atag + "CnRaw", f2(a_ob["cnceleb_e"]["eer_percent"])))
            nums.append(macro_def(atag + "CnQmf", f2(a_ob["cnceleb_e"]["qmf"]["eer_percent"])))
            nums.append(macro_def(atag + "CnMulti", f2(a_ob["cnceleb_e"]["multi_enroll_map"]["as_norm"]["eer_percent"])))
            nums.append(macro_def(atag + "LsRaw", f2(a_ob["librispeech_test_clean"]["eer_percent"])))
            nums.append(macro_def(atag + "LsQmf", f2(a_ob["librispeech_test_clean"]["qmf"]["eer_percent"])))
            nums.append(macro_def(atag + "LsMulti", f2(a_ob["librispeech_test_clean"]["multi_enroll_3utt"]["as_norm"]["eer_percent"])))
            nums.append(macro_def(atag + "PooledOne", f2(d1(arm_d, "universal_pooled")["eer_percent"])))
            nums.append(macro_def(atag + "PooledThree", f2(d3(arm_d, "universal_pooled")["eer_percent"])))
            nums.append(macro_def(atag + "MacroOne", f2(macro(arm_d, 1)["eer_percent"])))
            nums.append(macro_def(atag + "MacroThree", f2(macro(arm_d, 3)["eer_percent"])))

    pw = pfas_sweep["winner"]
    nums.append(macro_def("pfasSweepNumCands", str(len(pfas_sweep["candidates"]))))
    nums.append(macro_def("pfasSweepWarmup", str(int(pfas_sweep["head_warmup_steps"]))))
    nums.append(macro_def("pfasSweepJoint", str(int(pfas_sweep["joint_finetune_steps"]))))
    nums.append(
        macro_def(
            "pfasSweepSteps",
            str(int(pfas_sweep["head_warmup_steps"]) + int(pfas_sweep["joint_finetune_steps"])),
        )
    )
    nums.append(macro_def("pfasWinnerValOne", f2(pw["val_1utt_eer_percent"])))
    nums.append(macro_def("pfasWinnerValThree", f2(pw["val_3utt_eer_percent"])))
    nums.append(macro_def("pfasWinnerValComp", f2(pw["composite_val_eer_percent"])))
    nums.append(macro_def("pfasRankFourComp", f2(pfas_sweep["ranking"][3]["composite_val_eer_percent"])))
    nums.append(macro_def("pfasRankSevenComp", f2(pfas_sweep["ranking"][6]["composite_val_eer_percent"])))
    nums.append(macro_def("pfasWorstValComp", f2(pfas_sweep["ranking"][-1]["composite_val_eer_percent"])))

    # Phase-1 pilot summary macros.
    p1_cap = p1_pilot["capacity_scaling_comparison"]["best_converged"]
    nums.append(macro_def("pOneBaseLsBest", f2(p1_cap["base"]["librispeech_1utt_eer"])))
    nums.append(macro_def("pOneSmallLsBest", f2(p1_cap["small"]["librispeech_1utt_eer"])))
    nums.append(macro_def("pOneTinyLsBest", f2(p1_cap["tiny"]["librispeech_1utt_eer"])))
    p1_mw = p1_pilot["margin_warmup_comparison"]["univ_conformer_small_arcface"]
    nums.append(macro_def("pOneArcConstPooled", f2(p1_mw["constant_margin_best_pooled_1utt_eer"])))
    nums.append(macro_def("pOneArcWarmPooled", f2(p1_mw["warmup500_best_pooled_1utt_eer"])))

    # Matched univ_* vs ls_* pairs (19 Tier-1 controlled pairs).
    pairs = []
    for mid in univ_order:
        twin = "ls_" + mid[len("univ_"):]
        if twin in controlled_ex:
            pairs.append((mid, twin))
    worse_macro1 = [m for m, t in pairs if macro(ex[m], 1)["eer_percent"] > macro(ex[t], 1)["eer_percent"]]
    worse_pool1 = [
        m for m, t in pairs
        if d1(ex[m], "universal_pooled")["eer_percent"] > d1(ex[t], "universal_pooled")["eer_percent"]
    ]
    worse_pool3 = [
        m for m, t in pairs
        if d3(ex[m], "universal_pooled")["eer_percent"] > d3(ex[t], "universal_pooled")["eer_percent"]
    ]
    nums.append(macro_def("numMatchedPairs", str(len(pairs))))
    nums.append(macro_def("univWorseMacroOne", str(len(worse_macro1))))
    nums.append(macro_def("univWorsePooledOne", str(len(worse_pool1))))
    nums.append(macro_def("univWorsePooledThree", str(len(worse_pool3))))
    nums.append(macro_def("univWinsPooledOne", str(len(pairs) - len(worse_pool1))))
    nums.append(macro_def("univWinsMacroOne", str(len(pairs) - len(worse_macro1))))
    nums.append(macro_def("univWinsPooledThree", str(len(pairs) - len(worse_pool3))))
    pool1_gains = [
        d1(ex[t], "universal_pooled")["eer_percent"]
        - d1(ex[m], "universal_pooled")["eer_percent"]
        for m, t in pairs
    ]
    macro1_gains = [
        macro(ex[t], 1)["eer_percent"] - macro(ex[m], 1)["eer_percent"]
        for m, t in pairs
    ]
    pool3_gains = [
        d3(ex[t], "universal_pooled")["eer_percent"]
        - d3(ex[m], "universal_pooled")["eer_percent"]
        for m, t in pairs
    ]
    nums.append(macro_def("univGainPooledOneMin", f2(min(pool1_gains))))
    nums.append(macro_def("univGainPooledOneMax", f2(max(pool1_gains))))
    nums.append(macro_def("univGainMacroOneMin", f2(min(macro1_gains))))
    nums.append(macro_def("univGainMacroOneMax", f2(max(macro1_gains))))
    nums.append(macro_def("univGainPooledThreeMin", f2(min(pool3_gains))))
    nums.append(macro_def("univGainPooledThreeMax", f2(max(pool3_gains))))

    def pair_label(mid: str) -> str:
        e = ex[mid]
        backbone = {
            "conformer": "Conformer",
            "transformer": "Transformer",
            "mamba": "Mamba",
            "lstm": "LSTM",
            "ecapa_tdnn": "ECAPA-TDNN",
            "resnet": "ResNet-18",
        }.get(e["backbone"], e["backbone"])
        obj = {
            "ge2e_softmax": "GE2E",
            "ge2e_contrast": "GE2E-C$^{*}$",
            "extended_set_softmax": "BXS$^{*}$",
            "arcface": "ArcFace",
            "cosface": "CosFace",
            "sphereface": "SphereFace",
            "triplet": "triplet",
        }.get(e["loss_type"], e["loss_type"])
        if e["scoring_type"] == "dr_vector":
            obj = "Dr-Vec$^{*}$"
        elif e["scoring_type"] == "pfas":
            obj = "PFAS-SN$^{*}$/\\allowbreak " + ("GE2E" if e["loss_type"] == "ge2e_softmax" else "BXS")
        return f"{backbone}-{e['size_variant'][0].upper()}/\\allowbreak {obj}"

    # VoxCeleb1-O subset summary + Spearman vs pooled (over all 49 models and over the 44 Tier-1 models).
    vox_eers_44 = {k: v["eer_percent"] for k, v in vox["models"].items() if k in controlled_ex}
    vox_eers_all = {k: v["eer_percent"] for k, v in vox["models"].items() if k in ex}
    pooled_44 = {k: d1(ex[k], "universal_pooled")["eer_percent"] for k in vox_eers_44}
    pooled_all_vox = {k: d1(ex[k], "universal_pooled")["eer_percent"] for k in vox_eers_all}
    from scipy import stats as sstats

    rho, pval = sstats.spearmanr([pooled_44[k] for k in vox_eers_44], [vox_eers_44[k] for k in vox_eers_44])
    vox_in = {k: d1(ex[k], "voxceleb1")["eer_percent"] for k in vox_eers_44}
    rho_v, pval_v = sstats.spearmanr([vox_in[k] for k in vox_eers_44], [vox_eers_44[k] for k in vox_eers_44])
    rho_all, pval_all = sstats.spearmanr([pooled_all_vox[k] for k in vox_eers_all], [vox_eers_all[k] for k in vox_eers_all])
    vox_in_all = {k: d1(ex[k], "voxceleb1")["eer_percent"] for k in vox_eers_all}
    rho_v_all, pval_v_all = sstats.spearmanr([vox_in_all[k] for k in vox_eers_all], [vox_eers_all[k] for k in vox_eers_all])
    best_vox = min(vox_eers_44.items(), key=lambda kv: kv[1])
    worst_vox = max(vox_eers_44.items(), key=lambda kv: kv[1])
    best_vox_all = min(vox_eers_all.items(), key=lambda kv: kv[1])
    nums.append(macro_def("voxSubsetTrials", f"{vox['num_trials']:,}"))
    nums.append(macro_def("voxSubsetTarget", f"{vox['num_target_trials']:,}"))
    nums.append(macro_def("voxSubsetImpostor", f"{vox['num_impostor_trials']:,}"))
    nums.append(macro_def("voxSubsetTargetPct", f"{100.0 * vox['num_target_trials'] / vox['num_trials']:.1f}"))
    nums.append(macro_def("voxSubsetUtts", f"{vox['num_unique_utterances']:,}"))
    nums.append(macro_def("voxSubsetSpeakers", str(len(vox["eval_speakers"]))))
    nums.append(macro_def("voxSubsetSpeakerList", ", ".join(tex_escape(s) for s in vox["eval_speakers"])))
    nums.append(macro_def("voxSubsetVadDb", f"{vox['vad_energy_db_threshold']:.0f}"))
    nums.append(macro_def("voxSubsetMinEer", f2(best_vox[1])))
    nums.append(macro_def("voxSubsetMinModel", tex_escape(best_vox[0])))
    nums.append(macro_def("voxSubsetAllMinEer", f2(best_vox_all[1])))
    nums.append(macro_def("voxSubsetAllMinModel", tex_escape(best_vox_all[0])))
    nums.append(macro_def("voxSubsetMaxEer", f2(worst_vox[1])))
    nums.append(macro_def("voxSubsetMaxModel", tex_escape(worst_vox[0])))
    nums.append(macro_def("spearmanPooledVox", f"{rho:.2f}"))
    nums.append(macro_def("spearmanPooledVoxP", fmt_sci(pval)))
    nums.append(macro_def("spearmanInVoxVox", f"{rho_v:.2f}"))
    nums.append(macro_def("spearmanInVoxVoxP", fmt_sci(pval_v)))
    nums.append(macro_def("spearmanAllPooledVox", f"{rho_all:.2f}"))
    nums.append(macro_def("spearmanAllPooledVoxP", fmt_sci(pval_all)))
    nums.append(macro_def("spearmanAllInVoxVox", f"{rho_v_all:.2f}"))
    nums.append(macro_def("spearmanAllInVoxVoxP", fmt_sci(pval_v_all)))

    # Ablations (all converged v0.5.0).
    st = abl.get("studies", {})
    dm = abl["domain_mixing_ablation"]
    for k, tag in (("p_within_0.0", "mixZero"), ("p_within_0.75", "mixDefault"), ("p_within_1.0", "mixOne")):
        add_mde_nums(nums, tag, dm[k]["multi_domain_eval"])

    xs = abl["extended_set_and_dr_ablation"]
    add_mde_nums(nums, "xsPure", xs["ext_softmax_pure_eq3"]["multi_domain_eval"])
    add_mde_nums(nums, "xsBlend", xs["ext_softmax_blended"]["multi_domain_eval"])
    add_mde_nums(nums, "drThreePure", xs["dr_vectors_3term_pure_eq7"]["multi_domain_eval"])
    add_mde_nums(nums, "drOneBlend", xs["dr_vectors_1term_blended"]["multi_domain_eval"])
    if "dr_vector_terms_and_contrast_ablation" in st:
        dvc = st["dr_vector_terms_and_contrast_ablation"]
        add_mde_nums(nums, "drThreeSymBlend", dvc["dr_vector_3term_sym_skip"]["multi_domain_eval"])
    if "pfas_sn_attention_and_anchor_ablation" in st:
        pfa = st["pfas_sn_attention_and_anchor_ablation"]
        add_mde_nums(nums, "pfasMlpAnchor", pfa["pfas_sn_mlp_with_anchor"]["multi_domain_eval"])
        add_mde_nums(nums, "pfasMlpNoAnchor", pfa["pfas_sn_mlp_no_anchor"]["multi_domain_eval"])
        add_mde_nums(nums, "pfasDotAnchor", pfa["pfas_sn_dot_with_anchor"]["multi_domain_eval"])
        add_mde_nums(nums, "pfasMaxpoolAnchor", pfa["pfas_sn_maxpool_with_anchor"]["multi_domain_eval"])

    vl = abl["variable_length_ablation"]
    for k, tag in (("variable_length_true", "vlVar"), ("variable_length_false", "vlFixed")):
        add_mde_nums(nums, tag + "Full", vl[k]["full_seq_eval"])
        add_mde_nums(nums, tag + "Short", vl[k]["short_1s_eval"])

    own_h = r2.get("own_schedule_horizon", {})
    for k, tag in (
        ("steps_400", "ownFour"),
        ("steps_800", "ownEight"),
        ("steps_1200", "ownTwelve"),
        ("steps_2400", "ownTwentyFour"),
        ("steps_4600", "ownFortySix"),
    ):
        if k in own_h:
            add_mde_nums(nums, tag, own_h[k]["multi_domain_eval"])

    m_ls = abl["mamba_directionality_ablation"]
    for k, tag in (("unidirectional", "mambaUni"), ("bidirectional", "mambaBi")):
        add_mde_nums(nums, tag, m_ls[k]["multi_domain_eval"])
    if "mamba_direction_ablation" in st:
        m_u = st["mamba_direction_ablation"]
        add_mde_nums(nums, "mambaUnivUni", m_u["mamba_unidirectional"]["multi_domain_eval"])
        add_mde_nums(nums, "mambaUnivBi", m_u["mamba_bidirectional"]["multi_domain_eval"])

    if "ecapa_norm_order_ablation" in st:
        ec = st["ecapa_norm_order_ablation"]
        add_mde_nums(nums, "ecapaLnBefore", ec["ecapa_ln_before_act"]["multi_domain_eval"])
        add_mde_nums(nums, "ecapaLnAfter", ec["ecapa_bn_style_after_act"]["multi_domain_eval"])

    if "vad_and_agc_ablation" in st:
        va = st["vad_and_agc_ablation"]
        add_mde_nums(nums, "vadDefault", va["vad_26db_with_agc"]["multi_domain_eval"])
        add_mde_nums(nums, "vadNoAgc", va["vad_26db_no_agc"]["multi_domain_eval"])
        add_mde_nums(nums, "vadTwentyNoAgc", va["vad_20db_no_agc"]["multi_domain_eval"])

    if "paper_faithful_presets_smoke_test" in st:
        pf = st["paper_faithful_presets_smoke_test"]
        add_mde_nums(nums, "pfGe", pf["icassp2018_ge2e_pure_paper"]["multi_domain_eval"])
        add_mde_nums(nums, "pfDr", pf["interspeech2021_dr_vectors_pure_paper"]["multi_domain_eval"])
        add_mde_nums(nums, "pfAtp", pf["odyssey2022_conformer_atp_pure_paper"]["multi_domain_eval"])
        add_mde_nums(nums, "pfPfas", pf["odyssey2022_pfas_pure_paper"]["multi_domain_eval"])

    # Step-0 fingerprint histogram.
    fp_ck = fps.get("checkpoints", {})
    hist = fps.get("histogram", {})
    nums.append(macro_def("fpTotal", str(len(fp_ck))))
    nums.append(
        macro_def(
            "fpNumVariable",
            str(
                hist.get("variable", 0)
                + hist.get("variable_approx", 0)
                + hist.get("both_approx", 0)
            ),
        )
    )
    nums.append(macro_def("fpNumVariableExact", str(hist.get("variable", 0))))
    nums.append(macro_def("fpNumVariableApprox", str(hist.get("variable_approx", 0))))
    nums.append(macro_def("fpNumBothApprox", str(hist.get("both_approx", 0))))
    nums.append(macro_def("fpNumBoth", str(hist.get("both", 0))))
    nums.append(macro_def("fpNumFixedAll", str(hist.get("fixed", 0) + hist.get("fixed_approx", 0))))
    nums.append(macro_def("fpNumNeither", str(hist.get("neither", 0))))
    nums.append(macro_def("fpCommit", tex_escape(str(fps.get("code_commit", "")))))
    nums.append(
        macro_def(
            "fpBothList",
            ", ".join("\\texttt{" + tex_escape(k) + "}" for k, v in sorted(fp_ck.items()) if v["match"] == "both"),
        )
    )

    # TFLite (6 Tier-1 + 3 Tier-2 models = 18 binaries).
    tf = bench["tflite_exports"]
    ratios, fp32_mb, int8_mb, cos = [], [], [], []
    tier1_fp32_mb, tier1_int8_mb = [], []
    for k, v in tf.items():
        if k.endswith("_int8"):
            base = k[:-len("_int8")]
            fp = tf[base + "_fp32"]["size_bytes"]
            ratios.append(fp / v["size_bytes"])
            fp32_mb.append(fp / 1e6)
            int8_mb.append(v["size_bytes"] / 1e6)
            if not base.startswith("univ_full_"):
                tier1_fp32_mb.append(fp / 1e6)
                tier1_int8_mb.append(v["size_bytes"] / 1e6)
            cos.append(v["cosine_fidelity_vs_flax"])
    nums.append(macro_def("numTflite", str(len(tf))))
    nums.append(macro_def("numExportedModels", str(len(tf) // 2)))
    nums.append(macro_def("numTfliteTierOne", "12"))
    nums.append(macro_def("tfliteRatioMin", f"{min(ratios):.2f}"))
    nums.append(macro_def("tfliteRatioMax", f"{max(ratios):.2f}"))
    nums.append(macro_def("tfliteFpMinMB", f"{min(tier1_fp32_mb):.2f}"))
    nums.append(macro_def("tfliteFpMaxMB", f"{max(tier1_fp32_mb):.2f}"))
    nums.append(macro_def("tfliteAllFpMaxMB", f"{max(fp32_mb):.2f}"))
    nums.append(macro_def("tfliteIntMinMB", f"{min(tier1_int8_mb):.2f}"))
    nums.append(macro_def("tfliteIntMaxMB", f"{max(tier1_int8_mb):.2f}"))
    nums.append(macro_def("tfliteAllIntMaxMB", f"{max(int8_mb):.2f}"))
    nums.append(macro_def("tfliteCosMin", f"{min(cos):.6f}"))
    nums.append(macro_def("tfliteWindowFrames", "180"))
    nums.append(macro_def("tfliteWindowSec", "1.8"))

    tfl_tags = [
        ("tfliteUnivConf", "univ_conformer_small_ge2e"),
        ("tfliteUnivEcapa", "univ_ecapa_tdnn_small_ge2e"),
        ("tfliteUnivLstm", "univ_lstm_small_ge2e"),
        ("tfliteLsConf", "ls_conformer_small_ge2e"),
        ("tfliteLsEcapa", "ls_ecapa_tdnn_small_ge2e"),
        ("tfliteLsLstm", "ls_lstm_small_ge2e"),
    ]
    tfl_cos_means, tfl_cos_mins, tfl_eer_deltas, tfl_trunc = [], [], [], []
    for ftag_tfl, fmid in (
        ("tfliteConfL", "univ_full_conformer_large_arcface"),
        ("tfliteEcapaL", "univ_full_ecapa_c1024_arcface"),
        ("tfliteResnetL", "univ_full_resnet34_arcface"),
        ("tfliteEres", "univ_full_eres2conformer_large"),
        ("tfliteVtwo", "univ_full_eres2conformer_v2_large"),
        ("tfliteXl", "univ_full_eres2conformer_xl"),
        ("tfliteVtwoDiar", "univ_full_eres2conformer_v2_diar"),
        ("tfliteXlDiar", "univ_full_eres2conformer_xl_diar"),
        ("tfliteXlAdapt", "univ_full_eres2conformer_xl_diar_adapt"),
    ):
        fte = ex.get(fmid, {}).get("tflite_eval", {})
        if fte:
            nums.append(macro_def(ftag_tfl + "FpEer", f2(fte["eer_percent"])))
            nums.append(macro_def(ftag_tfl + "IntEer", f2(fte["eer_percent_int8"])))
            nums.append(macro_def(ftag_tfl + "CosMean", f"{fte['cos_fp32_int8_mean']:.6f}"))
            if "latency_ms_fp32_mean_50runs" in fte:
                nums.append(macro_def(ftag_tfl + "LatFp", f2(fte["latency_ms_fp32_mean_50runs"])))
                nums.append(macro_def(ftag_tfl + "LatInt", f2(fte["latency_ms_int8_mean_50runs"])))
            nums.append(
                macro_def(
                    ftag_tfl + "FpMB",
                    f"{tf[fmid + '_fp32']['size_bytes'] / 1e6:.2f}",
                )
            )
            nums.append(
                macro_def(
                    ftag_tfl + "IntMB",
                    f"{tf[fmid + '_int8']['size_bytes'] / 1e6:.2f}",
                )
            )
            tfl_cos_means.append(fte["cos_fp32_int8_mean"])
            if "cos_fp32_int8_min" in fte:
                tfl_cos_mins.append(fte["cos_fp32_int8_min"])
            tfl_eer_deltas.append(abs(fte["eer_percent_int8"] - fte["eer_percent"]))
    for ttag, tmid in tfl_tags:
        te = ex[tmid].get("tflite_eval", {})
        if te:
            nums.append(macro_def(ttag + "FpEer", f2(te["eer_percent"])))
            nums.append(macro_def(ttag + "IntEer", f2(te["eer_percent_int8"])))
            nums.append(macro_def(ttag + "JaxFullEer", f2(te["jax_full_seq_eer_percent"])))
            nums.append(macro_def(ttag + "CosMean", f"{te['cos_fp32_int8_mean']:.6f}"))
            nums.append(macro_def(ttag + "LatFp", f2(te["latency_ms_fp32_mean_50runs"])))
            nums.append(macro_def(ttag + "LatInt", f2(te["latency_ms_int8_mean_50runs"])))
            tfl_cos_means.append(te["cos_fp32_int8_mean"])
            tfl_cos_mins.append(te["cos_fp32_int8_min"])
            tfl_eer_deltas.append(abs(te["eer_percent_int8"] - te["eer_percent"]))
            tfl_trunc.append(te["eer_percent"] - te["jax_full_seq_eer_percent"])
    if tfl_cos_means:
        nums.append(macro_def("tfliteCosMeanMin", f"{min(tfl_cos_means):.6f}"))
        nums.append(macro_def("tfliteCosMeanMax", f"{max(tfl_cos_means):.6f}"))
        nums.append(macro_def("tfliteCosUttMin", f"{min(tfl_cos_mins):.6f}"))
        nums.append(macro_def("tfliteEerDeltaMax", f2(max(tfl_eer_deltas))))
        nums.append(macro_def("tfliteTruncMin", f2(min(tfl_trunc))))
        nums.append(macro_def("tfliteTruncMax", f2(max(tfl_trunc))))
        lat_u = ex["univ_lstm_small_ge2e"]["tflite_eval"]
        lat_l = ex["ls_lstm_small_ge2e"]["tflite_eval"]
        sp_u = lat_u["latency_ms_fp32_mean_50runs"] / lat_u["latency_ms_int8_mean_50runs"]
        sp_l = lat_l["latency_ms_fp32_mean_50runs"] / lat_l["latency_ms_int8_mean_50runs"]
        nums.append(macro_def("tfliteLstmSpeedupMin", f"{min(sp_u, sp_l):.1f}"))
        nums.append(macro_def("tfliteLstmSpeedupMax", f"{max(sp_u, sp_l):.1f}"))

    # Multi-seed stability (all 3 models in r2["multi_seed_stability"]).
    ms_all = r2.get("multi_seed_stability", {})
    seed_tag_map = {
        "univ_conformer_small_ge2e": "univSeed",
        "univ_ecapa_tdnn_small_ge2e": "ecapaSeed",
        "ls_conformer_small_ge2e": "lsSeed",
    }
    for mid, stag in seed_tag_map.items():
        if mid not in ms_all:
            continue
        mv = ms_all[mid]
        s1, s3 = mv["summary_1utt"], mv["summary_3utt"]
        per = sorted(mv["per_seed"], key=lambda r: int(r["seed"]))
        nums.append(macro_def(stag + "N", str(len(per))))
        nums.append(macro_def(stag + "Seeds", ", ".join(str(r["seed"]) for r in per)))
        nums.append(macro_def(stag + "PooledOneList", " / ".join(f2(r["summary"]["pooled_1utt_eer"]) for r in per)))
        for dom_k, sum_k, ktag in (
            ("universal_pooled", "pooled_1utt_eer", "PooledOne"),
            ("macro_avg", "macro_1utt_eer", "MacroOne"),
            ("librispeech", "ls_1utt_eer", "LsOne"),
        ):
            s_vals = [float(r["summary"][sum_k]) for r in per]
            nums.append(macro_def(stag + ktag + "Mean", f2(s1[dom_k]["mean"])))
            nums.append(macro_def(stag + ktag + "Std", f2(s1[dom_k]["std"])))
            nums.append(macro_def(stag + ktag + "Min", f2(min(s_vals))))
            nums.append(macro_def(stag + ktag + "Max", f2(max(s_vals))))
        for dom_k, ktag in (("universal_pooled", "PooledThree"), ("macro_avg", "MacroThree")):
            nums.append(macro_def(stag + ktag + "Mean", f2(s3[dom_k]["mean"])))
            nums.append(macro_def(stag + ktag + "Std", f2(s3[dom_k]["std"])))

    # Matched-data controls (2,400 steps and 4,600 steps).
    md_ctrl = r2.get("matched_data_control", {})
    if "univ_subsampled_5788_2400steps" in md_ctrl:
        m24 = md_ctrl["univ_subsampled_5788_2400steps"]
        nums.append(macro_def("matchedUtts", f"{m24['num_train_utterances']:,}"))
        nums.append(macro_def("matchedSpeakers", str(m24["num_train_speakers"])))
        add_mde_nums(nums, "matchedTwentyFour", m24["multi_domain_eval"])
        ls_p1_raw = d1(ex["ls_conformer_small_ge2e"], "universal_pooled")["eer_percent"]
        m24_p1_raw = m24["multi_domain_eval"]["domains_1utt"]["universal_pooled"]["eer_percent"]
        nums.append(macro_def("matchedGainPooledOne", f2(ls_p1_raw - m24_p1_raw)))
    if "univ_subsampled_5788_4600steps" in md_ctrl:
        m46 = md_ctrl["univ_subsampled_5788_4600steps"]
        add_mde_nums(nums, "matchedFortySix", m46["multi_domain_eval"])

    emit(os.path.join(out_dir, "numbers.tex"), "".join(nums))

    # ------------------------------------------------------------ Table: highlights
    highlight_order = HIGHLIGHT_SPEC_ORDER + HIGHLIGHT_UNIV_ORDER
    rows = []
    for mid in highlight_order:
        e = ex[mid]
        rows.append((
            mid,
            int(e["num_steps"]),
            int(e["num_parameters"]) / 1e6,
            d1(e, "librispeech")["eer_percent"],
            d3(e, "librispeech")["eer_percent"],
            d1(e, "cnceleb2")["eer_percent"],
            d3(e, "cnceleb2")["eer_percent"],
            d1(e, "voxceleb1")["eer_percent"],
            d3(e, "voxceleb1")["eer_percent"],
            macro(e, 1)["eer_percent"],
            macro(e, 3)["eer_percent"],
            d1(e, "universal_pooled")["eer_percent"],
            d3(e, "universal_pooled")["eer_percent"],
        ))
    cols = list(zip(*rows))
    bolded = {i: bold_min(list(cols[i])) for i in range(3, 13)}
    lines = []
    for r_i, r in enumerate(rows):
        cells = [f"\\texttt{{{tex_escape(r[0])}}}", f"{r[1]:,}", f"{r[2]:.2f}"]
        for i in (3, 5, 7):
            cells.append(bolded[i][r_i])
        for i in (9, 11):
            cells.append(f"{bolded[i][r_i]} / {bolded[i + 1][r_i]}")
        lines.append(" & ".join(cells) + " \\\\")
        if r_i == len(HIGHLIGHT_SPEC_ORDER) - 1:
            lines.append("\\midrule")
    tbl = (
        "\\begin{tabular}{@{}lrrccccc@{}}\n\\toprule\n"
        "\\textbf{Model ID (\\texttt{wq2012/*})} & \\textbf{Steps} & \\textbf{M par.} & "
        "\\textbf{LS 1-utt} & \\textbf{CN 1-utt} & \\textbf{VOX 1-utt} & "
        "\\textbf{Macro 1 / 3-utt} & \\textbf{Pooled 1 / 3-utt} \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    emit(os.path.join(out_dir, "table_highlights.tex"), tbl)

    # ------------------------------------ Table: Tier-2 Full-Data Flagships & Fusion
    min_vx_raw = min(round(fm[mid]["official_benchmarks"]["voxceleb1_o"]["eer_percent"], 2) for mid in FLAGSHIP_ORDER)
    min_vx_asn = min(
        round(fm[mid]["official_benchmarks"]["voxceleb1_o"]["as_norm"]["eer_percent"], 2)
        for mid in FLAGSHIP_ORDER
        if "as_norm" in fm[mid]["official_benchmarks"]["voxceleb1_o"]
    )
    min_vx_qmf = min(round(fm[mid]["official_benchmarks"]["voxceleb1_o"]["qmf"]["eer_percent"], 2) for mid in FLAGSHIP_ORDER)
    min_vx_dcf_raw = min(round(fm[mid]["official_benchmarks"]["voxceleb1_o"]["min_dcf_001"], 4) for mid in FLAGSHIP_ORDER)
    min_vx_dcf_qmf = min(round(fm[mid]["official_benchmarks"]["voxceleb1_o"]["qmf"]["min_dcf_001"], 4) for mid in FLAGSHIP_ORDER)
    min_cn_qmf = min(round(fm[mid]["official_benchmarks"]["cnceleb_e"]["qmf"]["eer_percent"], 2) for mid in FLAGSHIP_ORDER)
    min_cn_me = min(
        round(
            fm[mid]["official_benchmarks"]["cnceleb_e"]["multi_enroll_map"].get(
                "as_norm", fm[mid]["official_benchmarks"]["cnceleb_e"]["multi_enroll_map"]
            )["eer_percent"],
            2,
        )
        for mid in FLAGSHIP_ORDER
    )
    min_ls_qmf = min(round(fm[mid]["official_benchmarks"]["librispeech_test_clean"]["qmf"]["eer_percent"], 2) for mid in FLAGSHIP_ORDER)
    min_ls_3 = min(
        round(
            fm[mid]["official_benchmarks"]["librispeech_test_clean"]["multi_enroll_3utt"].get(
                "as_norm", fm[mid]["official_benchmarks"]["librispeech_test_clean"]["multi_enroll_3utt"]
            )["eer_percent"],
            2,
        )
        for mid in FLAGSHIP_ORDER
    )
    min_p1 = min(round(d1(fm[mid], "universal_pooled")["eer_percent"], 2) for mid in FLAGSHIP_ORDER)
    min_p3 = min(round(d3(fm[mid], "universal_pooled")["eer_percent"], 2) for mid in FLAGSHIP_ORDER)
    flines = []
    for mid in FLAGSHIP_ORDER:
        m = fm[mid]
        ob = m["official_benchmarks"]
        vx, cn, ls = ob["voxceleb1_o"], ob["cnceleb_e"], ob["librispeech_test_clean"]
        vx_raw_s = f"\\textbf{{{f2(vx['eer_percent'])}}}" if round(vx["eer_percent"], 2) == min_vx_raw else f2(vx["eer_percent"])
        if "as_norm" in vx:
            vx_as = f"\\textbf{{{f2(vx['as_norm']['eer_percent'])}}}" if round(vx["as_norm"]["eer_percent"], 2) == min_vx_asn else f2(vx["as_norm"]["eer_percent"])
        else:
            vx_as = "---"
        vx_qmf = f"\\textbf{{{f2(vx['qmf']['eer_percent'])}}}" if round(vx["qmf"]["eer_percent"], 2) == min_vx_qmf else f2(vx["qmf"]["eer_percent"])
        vx_dcf_r = f"\\textbf{{{vx['min_dcf_001']:.3f}}}" if round(vx["min_dcf_001"], 4) == min_vx_dcf_raw else f"{vx['min_dcf_001']:.3f}"
        vx_dcf_q = f"\\textbf{{{vx['qmf']['min_dcf_001']:.3f}}}" if round(vx["qmf"]["min_dcf_001"], 4) == min_vx_dcf_qmf else f"{vx['qmf']['min_dcf_001']:.3f}"
        cn_qmf = f"\\textbf{{{f2(cn['qmf']['eer_percent'])}}}" if round(cn["qmf"]["eer_percent"], 2) == min_cn_qmf else f2(cn["qmf"]["eer_percent"])
        cn_me_val = cn["multi_enroll_map"].get("as_norm", cn["multi_enroll_map"])["eer_percent"]
        cn_me_s = f"\\textbf{{{f2(cn_me_val)}}}" if round(cn_me_val, 2) == min_cn_me else f2(cn_me_val)
        ls_qmf = f"\\textbf{{{f2(ls['qmf']['eer_percent'])}}}" if round(ls["qmf"]["eer_percent"], 2) == min_ls_qmf else f2(ls["qmf"]["eer_percent"])
        ls_3_val = ls["multi_enroll_3utt"].get("as_norm", ls["multi_enroll_3utt"])["eer_percent"]
        ls_3 = f"\\textbf{{{f2(ls_3_val)}}}" if round(ls_3_val, 2) == min_ls_3 else f2(ls_3_val)
        p1_v = d1(m, "universal_pooled")["eer_percent"]
        p3_raw = d3(m, "universal_pooled")["eer_percent"]
        p1_s = f"\\textbf{{{f2(p1_v)}}}" if round(p1_v, 2) == min_p1 else f2(p1_v)
        p3_s = f"\\textbf{{{f2(p3_raw)}}}" if round(p3_raw, 2) == min_p3 else f2(p3_raw)
        short_id = mid[len("univ_full_"):] if mid.startswith("univ_full_") else mid
        name = f"\\href{{https://huggingface.co/wq2012/{mid}}}{{\\texttt{{{tex_escape(short_id)}}}}}"
        if mid in TFLITE_REPOS:
            name += "$^{\\dagger}$"
        utts_m = int(m.get("dataset_stats", {}).get("total_utterances", 0)) / 1e6
        utts_s = f"{utts_m:.2f}" if utts_m > 0 else "---"
        flines.append(
            f"{name} & {utts_s} & {int(m['num_parameters'])/1e6:.2f} & {int(m['num_steps'])//1000}k & "
            f"{vx_raw_s}/{vx_as}/{vx_qmf} & {vx_dcf_r}/{vx_dcf_q} & "
            f"{f2(cn['eer_percent'])}/{cn_qmf}/{cn_me_s} & "
            f"{f2(ls['eer_percent'])}/{ls_qmf}/{ls_3} & "
            f"{p1_s}/{p3_s} \\\\"
        )
    flines.append("\\midrule")
    p2_m = (int(fm["univ_full_conformer_large_arcface"]["num_parameters"]) + int(fm["univ_full_ecapa_c1024_arcface"]["num_parameters"])) / 1e6
    act_mids = fus.get("models", FLAGSHIP_ORDER)
    all_act_params = sum(int(fm[m]["num_parameters"]) for m in act_mids if m in fm) / 1e6
    if "two_way_official_benchmarks" in fus:
        fus_2w = fus["two_way_official_benchmarks"]
        w2 = fus.get("two_way_weights_selected_on_val", [0.5, 0.5])
        fus_cands = [fus_2w, fus2]
        if "three_way_backbone_official_benchmarks" in fus and "univ_full_resnet34_arcface" in fm:
            fus_cands.append(fus["three_way_backbone_official_benchmarks"])
        if "four_way_backbone_official_benchmarks" in fus and "univ_full_eres2conformer_large" in fm:
            fus_cands.append(fus["four_way_backbone_official_benchmarks"])
        if "six_way_cosine_official_benchmarks" in fus and "univ_full_eres2conformer_xl" in fm:
            fus_cands.append(fus["six_way_cosine_official_benchmarks"])
        if "v080_7corpus_official_benchmarks" in fus:
            fus_cands.append(fus["v080_7corpus_official_benchmarks"])
        min_f_vx_as = min(round(fc["voxceleb1_o"]["eer_percent"], 2) for fc in fus_cands)
        min_f_vx_qmf = min(round(fc["voxceleb1_o"]["qmf"]["eer_percent"], 2) for fc in fus_cands)
        min_f_vx_dcf_as = min(round(fc["voxceleb1_o"]["min_dcf_001"], 4) for fc in fus_cands)
        min_f_vx_dcf_qmf = min(round(fc["voxceleb1_o"]["qmf"]["min_dcf_001"], 4) for fc in fus_cands)
        min_f_cn_qmf = min(round(fc["cnceleb_e"]["qmf"]["eer_percent"], 2) for fc in fus_cands)
        min_f_ls_qmf = min(round(fc["librispeech_test_clean"]["qmf"]["eer_percent"], 2) for fc in fus_cands)

        def fmt_tex_fus(fc: dict) -> tuple[str, str, str, str]:
            va = fc["voxceleb1_o"]["eer_percent"]
            vq = fc["voxceleb1_o"]["qmf"]["eer_percent"]
            da = fc["voxceleb1_o"]["min_dcf_001"]
            dq = fc["voxceleb1_o"]["qmf"]["min_dcf_001"]
            cq = fc["cnceleb_e"]["qmf"]["eer_percent"]
            lq = fc["librispeech_test_clean"]["qmf"]["eer_percent"]
            va_s = f"\\textbf{{{f2(va)}}}" if round(va, 2) == min_f_vx_as else f2(va)
            vq_s = f"\\textbf{{{f2(vq)}}}" if round(vq, 2) == min_f_vx_qmf else f2(vq)
            da_s = f"\\textbf{{{da:.3f}}}" if round(da, 4) == min_f_vx_dcf_as else f"{da:.3f}"
            dq_s = f"\\textbf{{{dq:.3f}}}" if round(dq, 4) == min_f_vx_dcf_qmf else f"{dq:.3f}"
            cq_s = f"\\textbf{{{f2(cq)}}}" if round(cq, 2) == min_f_cn_qmf else f2(cq)
            lq_s = f"\\textbf{{{f2(lq)}}}" if round(lq, 2) == min_f_ls_qmf and round(lq, 2) <= min_ls_qmf else f2(lq)
            return f"---/{va_s}/{vq_s}", f"{da_s}/{dq_s}", f"---/{cq_s}/---", f"---/{lq_s}/---"

        t2_vx, t2_dcf, t2_cn, t2_ls = fmt_tex_fus(fus_2w)
        flines.append(
            f"2-Way (${w2[0]:.2g}\\,\\text{{Conf-L}} + {w2[1]:.2g}\\,\\text{{ECAPA}}$) & 0.46 & {p2_m:.2f} & --- & "
            f"{t2_vx} & {t2_dcf} & {t2_cn} & {t2_ls} & --- \\\\"
        )
        if "three_way_backbone_official_benchmarks" in fus and "univ_full_resnet34_arcface" in fm:
            fus_3bb = fus["three_way_backbone_official_benchmarks"]
            p3bb_m = p2_m + int(fm["univ_full_resnet34_arcface"]["num_parameters"]) / 1e6
            t3_vx, t3_dcf, t3_cn, t3_ls = fmt_tex_fus(fus_3bb)
            flines.append(
                f"3-Way Backbone (+ ResNet-34) & 0.46 & {p3bb_m:.2f} & --- & "
                f"{t3_vx} & {t3_dcf} & {t3_cn} & {t3_ls} & --- \\\\"
            )
        if "four_way_backbone_official_benchmarks" in fus and "univ_full_eres2conformer_large" in fm:
            fus_4bb = fus["four_way_backbone_official_benchmarks"]
            p4bb_m = (
                p2_m
                + int(fm["univ_full_resnet34_arcface"]["num_parameters"]) / 1e6
                + int(fm["univ_full_eres2conformer_large"]["num_parameters"]) / 1e6
            )
            t4_vx, t4_dcf, t4_cn, t4_ls = fmt_tex_fus(fus_4bb)
            flines.append(
                f"4-Way Backbone (+ ERes2Conf-L) & $\\le$0.97 & {p4bb_m:.2f} & --- & "
                f"{t4_vx} & {t4_dcf} & {t4_cn} & {t4_ls} & --- \\\\"
            )
        if "six_way_cosine_official_benchmarks" in fus and "univ_full_eres2conformer_xl" in fm:
            fus_6c = fus["six_way_cosine_official_benchmarks"]
            cos_mids = fus.get("six_way_cosine_models", [])
            p6c_m = sum(int(fm[m]["num_parameters"]) for m in cos_mids if m in fm) / 1e6
            t6c_vx, t6c_dcf, t6c_cn, t6c_ls = fmt_tex_fus(fus_6c)
            flines.append(
                f"6-Way Cosine (+ V2-L + XL) & $\\le$1.71 & {p6c_m:.2f} & --- & "
                f"{t6c_vx} & {t6c_dcf} & {t6c_cn} & {t6c_ls} & --- \\\\"
            )
        if "univ_full_eres2conformer_xl_pfas" in act_mids:
            f_lbl = "6-Way Flagship (Cos + HybPFAS)"
        elif len(act_mids) == 6:
            f_lbl = "6-Way Flagship Fusion"
        elif len(act_mids) == 5:
            f_lbl = "5-Way Flagship Fusion"
        else:
            f_lbl = f"{len(act_mids)}-Way Flagship Fusion"
        t5_vx, t5_dcf, t5_cn, t5_ls = fmt_tex_fus(fus2)
        flines.append(
            f"{f_lbl} & $\\le$1.71 & {all_act_params:.2f} & --- & "
            f"{t5_vx} & {t5_dcf} & {t5_cn} & {t5_ls} & --- \\\\"
        )
        if "v080_7corpus_official_benchmarks" in fus:
            fus_v8 = fus["v080_7corpus_official_benchmarks"]
            v8_mids = fus.get("v080_7corpus_models", [])
            pv8_m = sum(int(fm[m]["num_parameters"]) for m in v8_mids if m in fm) / 1e6
            tv8_vx, tv8_dcf, tv8_cn, tv8_ls = fmt_tex_fus(fus_v8)
            if "v080_7corpus_multi_enroll_fusion" in fus:
                v8_me = fus["v080_7corpus_multi_enroll_fusion"]
                v8_cn_me = v8_me["cnceleb_e_multi_enroll_map"]["eer_percent"]
                v8_ls_me = v8_me["librispeech_test_clean_3utt"]["eer_percent"]
                v8_cq = fus_v8["cnceleb_e"]["qmf"]["eer_percent"]
                v8_lq = fus_v8["librispeech_test_clean"]["qmf"]["eer_percent"]
                v8_cq_s = f"\\textbf{{{f2(v8_cq)}}}" if round(v8_cq, 2) == min_f_cn_qmf else f2(v8_cq)
                v8_lq_s = f"\\textbf{{{f2(v8_lq)}}}" if round(v8_lq, 2) == min_f_ls_qmf and round(v8_lq, 2) <= min_ls_qmf else f2(v8_lq)
                v8_cn_me_s = f"\\textbf{{{f2(v8_cn_me)}}}" if round(v8_cn_me, 2) <= min_cn_me else f2(v8_cn_me)
                v8_ls_me_s = f"\\textbf{{{f2(v8_ls_me)}}}" if round(v8_ls_me, 2) <= min_ls_3 else f2(v8_ls_me)
                tv8_cn = f"---/{v8_cq_s}/{v8_cn_me_s}"
                tv8_ls = f"---/{v8_lq_s}/{v8_ls_me_s}"
            flines.append(
                f"{len(v8_mids)}-Way 7-Corpus (Diar + HybPFAS) & 2.48 & {pv8_m:.2f} & --- & "
                f"{tv8_vx} & {tv8_dcf} & {tv8_cn} & {tv8_ls} & --- \\\\"
            )
    else:
        flines.append(
            f"2-Way Fusion (Conf-L + ECAPA) & 0.46 & {p2_m:.2f} & --- & "
            f"---/{f2(fus2['voxceleb1_o']['eer_percent'])}/\\textbf{{{f2(fus2['voxceleb1_o']['qmf']['eer_percent'])}}} & --- & "
            f"---/{f2(fus2['cnceleb_e']['qmf']['eer_percent'])}/--- & "
            f"---/{f2(fus2['librispeech_test_clean']['qmf']['eer_percent'])}/--- & --- \\\\"
        )
    me_mids = fus3.get("models", ["univ_full_conformer_large_arcface", "univ_full_ecapa_c1024_arcface", "univ_full_conformer_large_pfas"])
    me_params = sum(int(fm[m]["num_parameters"]) for m in me_mids if m in fm) / 1e6
    me_lbl = f"{len(me_mids)}-Way Multi-Enroll Fusion"
    me_cn_v = fus3["cnceleb_e_multi_enroll_map"]["eer_percent"]
    me_ls_v = fus3["librispeech_test_clean_3utt"]["eer_percent"]
    me_cn_tex = f"\\textbf{{{f2(me_cn_v)}}}" if round(me_cn_v, 2) <= min_cn_me else f2(me_cn_v)
    me_ls_tex = f"\\textbf{{{f2(me_ls_v)}}}" if round(me_ls_v, 2) <= min_ls_3 else f2(me_ls_v)
    flines.append(
        f"{me_lbl} & $\\le$1.71 & {me_params:.2f} & --- & "
        f"--- & --- & ---/---/{me_cn_tex} & "
        f"---/---/{me_ls_tex} & --- \\\\"
    )
    ftbl = (
        "\\begin{tabular}{@{}lcrrccccc@{}}\n\\toprule\n"
        "\\textbf{Flagship / Score Fusion} & \\textbf{Utts} & \\textbf{M par.} & \\textbf{Steps} & "
        "\\textbf{Vox1-O EER} & \\textbf{Vox1-O DCF} & \\textbf{CN-E EER} & "
        "\\textbf{LS-clean} & \\textbf{Pooled 1/3} \\\\\n"
        " & (M) & & & (Raw/ASN/QMF) & (Raw|ASN/QMF$^{\\sharp}$) & (Raw/QMF/Multi) & (Raw/QMF/3u) & (Raw) \\\\\n\\midrule\n"
        + "\n".join(flines) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    emit(os.path.join(out_dir, "table_flagship_official.tex"), ftbl)

    # ---------------------------------------- Table: 3-Arm ERes2Conformer Ablation
    if eres_abl is not None:
        elines = []
        arms_d = eres_abl["arms"]
        e_specs = (
            ("eres2conformer_catp_sgd", "ERes2Conformer + CATP (2 Conf., SGD)"),
            ("eres2net_noconf_catp_sgd", "ERes2Net + CATP (0 Conf., SGD)"),
            ("eres2conformer_catp_adamw", "ERes2Conformer + CATP (2 Conf., AdamW)"),
        )
        min_em3 = min(round(macro(arms_d[aid], 3)["eer_percent"], 2) for aid, _ in e_specs)
        for aid, albl in e_specs:
            ad = arms_d[aid]
            aob = ad["official_benchmarks"]
            avx, acn, als = aob["voxceleb1_o"], aob["cnceleb_e"], aob["librispeech_test_clean"]
            acn_me = acn["multi_enroll_map"]["as_norm"]["eer_percent"]
            als_3 = als["multi_enroll_3utt"]["as_norm"]["eer_percent"]
            ap1 = d1(ad, "universal_pooled")["eer_percent"]
            ap3 = d3(ad, "universal_pooled")["eer_percent"]
            am1 = macro(ad, 1)["eer_percent"]
            am3 = macro(ad, 3)["eer_percent"]
            is_win = aid == eres_abl["winner"]

            def _bw(val: float, win: bool = is_win) -> str:
                return f"\\textbf{{{f2(val)}}}" if win else f2(val)

            am3_s = f"\\textbf{{{f2(am3)}}}" if round(am3, 2) == min_em3 else f2(am3)
            elines.append(
                f"{albl} & {int(ad['num_parameters'])/1e6:.2f} & "
                f"{f2(avx['eer_percent'])}/{f2(avx['as_norm']['eer_percent'])}/{_bw(avx['qmf']['eer_percent'])} & "
                f"{f2(acn['eer_percent'])}/{_bw(acn['qmf']['eer_percent'])}/{_bw(acn_me)} & "
                f"{f2(als['eer_percent'])}/{_bw(als['qmf']['eer_percent'])}/{_bw(als_3)} & "
                f"{_bw(ap1)}/{_bw(ap3)} & {_bw(am1)}/{am3_s} \\\\"
            )
        etbl = (
            "\\begin{tabular}{@{}lrccccc@{}}\n\\toprule\n"
            "\\textbf{27,500-Step TPU v6p Arm (Batch 512)} & \\textbf{M par.} & "
            "\\textbf{Vox1-O} & \\textbf{CN-E} & "
            "\\textbf{LS-clean} & \\textbf{Pooled 1/3} & \\textbf{Macro 1/3} \\\\\n"
            " & & (Raw/ASN/QMF) & (Raw/QMF/Multi) & (Raw/QMF/3u) & (Raw) & (Raw) \\\\\n\\midrule\n"
            + "\n".join(elines) + "\n\\bottomrule\n\\end{tabular}\n"
        )
        emit(os.path.join(out_dir, "table_eres2conf_ablation.tex"), etbl)

    if v070_abl is not None:
        vlines = []
        v_arms_d = v070_abl["arms"]
        arm_specs = (
            ("ctrl_base_noprune", "1.\\ Base ERes2Conf-L (33k, unpruned LMFT)"),
            ("clean_lmft_medoid_k1", "2.\\ + Medoid $K_{\\text{sc}}{=}1$ collapse (33k)"),
            ("clean_lmft_keep_k3_swa_bn", "3.\\ + Clean-$K_{\\text{sc}}{=}3$ + SWA + Exact BN + QMF-v2"),
            ("inter_topk_chan_aug_k3", "4.\\ + Inter-TopK ($K_{\\text{neg}}{=}5, 0.06$) + NB/Tilt"),
            ("v2_large_fwse_mfa123", "5.\\ V2-Large (2D-fwSE + 3-MFA + $3{\\times}$Conf)"),
            ("xl_64m_clean_k3", "6.\\ XL-64M ($C_0{=}80, (3,4,8,3), 4{\\times}640$D Conf)"),
            ("hybrid_pfas_attentive_scoring", "7.\\ + Hybrid-PFAS ($12$k ep., frozen Large)"),
        )
        present_arms = [v_arms_d[aid] for aid, _ in arm_specs if aid in v_arms_d]
        min_a_vx_r = min(round(a["official_benchmarks"]["voxceleb1_o"]["eer_percent"], 2) for a in present_arms)
        min_a_vx_q = min(round(a["official_benchmarks"]["voxceleb1_o"]["qmf"]["eer_percent"], 2) for a in present_arms)
        min_a_cn_m = min(round(a["official_benchmarks"]["cnceleb_e"]["multi_enroll_map"]["as_norm"]["eer_percent"], 2) for a in present_arms)
        min_a_ls_q = min(round(a["official_benchmarks"]["librispeech_test_clean"]["qmf"]["eer_percent"], 2) for a in present_arms)
        min_a_ls_3 = min(round(a["official_benchmarks"]["librispeech_test_clean"]["multi_enroll_3utt"]["as_norm"]["eer_percent"], 2) for a in present_arms)
        min_a_p1 = min(round(d1(a, "universal_pooled")["eer_percent"], 2) for a in present_arms)
        min_a_p3 = min(round(d3(a, "universal_pooled")["eer_percent"], 2) for a in present_arms)
        for aid, albl in arm_specs:
            if aid not in v_arms_d:
                continue
            ad = v_arms_d[aid]
            aob = ad["official_benchmarks"]
            avx, acn, als = aob["voxceleb1_o"], aob["cnceleb_e"], aob["librispeech_test_clean"]
            acn_me = acn["multi_enroll_map"]["as_norm"]["eer_percent"]
            als_3 = als["multi_enroll_3utt"]["as_norm"]["eer_percent"]
            ap1 = d1(ad, "universal_pooled")["eer_percent"]
            ap3 = d3(ad, "universal_pooled")["eer_percent"]
            vr_s = f"\\textbf{{{f2(avx['eer_percent'])}}}" if round(avx["eer_percent"], 2) == min_a_vx_r else f2(avx["eer_percent"])
            vq_s = f"\\textbf{{{f2(avx['qmf']['eer_percent'])}}}" if round(avx["qmf"]["eer_percent"], 2) == min_a_vx_q else f2(avx["qmf"]["eer_percent"])
            cm_s = f"\\textbf{{{f2(acn_me)}}}" if round(acn_me, 2) == min_a_cn_m else f2(acn_me)
            lq_s = f"\\textbf{{{f2(als['qmf']['eer_percent'])}}}" if round(als["qmf"]["eer_percent"], 2) == min_a_ls_q else f2(als["qmf"]["eer_percent"])
            l3_s = f"\\textbf{{{f2(als_3)}}}" if round(als_3, 2) == min_a_ls_3 else f2(als_3)
            p1_s = f"\\textbf{{{f2(ap1)}}}" if round(ap1, 2) == min_a_p1 else f2(ap1)
            p3_s = f"\\textbf{{{f2(ap3)}}}" if round(ap3, 2) == min_a_p3 else f2(ap3)
            vlines.append(
                f"{albl} & {int(ad['num_parameters'])/1e6:.2f} & {int(ad['num_steps']):,} & "
                f"{vr_s}/{f2(avx['as_norm']['eer_percent'])}/{vq_s} & "
                f"{f2(acn['eer_percent'])}/{f2(acn['qmf']['eer_percent'])}/{cm_s} & "
                f"{f2(als['eer_percent'])}/{lq_s}/{l3_s} & "
                f"{p1_s}/{p3_s} \\\\"
            )
        lh_runs_t = v070_abl.get("long_horizon_recipe_comparison", {}).get("runs", {})
        if lh_runs_t:
            vlines.append("\\midrule")
            v2_c_vt = lh_runs_t["v2_large_clean_162k"]["validation_trajectory"][-1]
            v2_i_vt = lh_runs_t["v2_large_interaug_162k"]["validation_trajectory"][-1]
            xl_c_vt = lh_runs_t["xl_64m_clean_145k"]["validation_trajectory"][-1]
            xl_i_vt = lh_runs_t["xl_64m_interaug_145k"]["validation_trajectory"][-1]
            lh_specs = (
                (
                    "v2_large_clean_162k",
                    f"V2-Large (Clean; val 1u/comp ${f2(v2_c_vt['val_1utt_eer_percent'])}/{f2(v2_c_vt['val_composite_eer_percent'])}\\%$)",
                ),
                (
                    "v2_large_interaug_162k",
                    f"V2-Large (InterAug; val 1u/comp ${f2(v2_i_vt['val_1utt_eer_percent'])}/{f2(v2_i_vt['val_composite_eer_percent'])}\\%$)$^{{\\star}}$",
                ),
                (
                    "xl_64m_clean_145k",
                    f"XL-64M (Clean; val 1u/comp ${f2(xl_c_vt['val_1utt_eer_percent'])}/{f2(xl_c_vt['val_composite_eer_percent'])}\\%$)",
                ),
                (
                    "xl_64m_interaug_145k",
                    f"XL-64M (InterAug; val 1u/comp ${f2(xl_i_vt['val_1utt_eer_percent'])}/{f2(xl_i_vt['val_composite_eer_percent'])}\\%$)$^{{\\star}}$",
                ),
            )
            present_lh = [lh_runs_t[k] for k, _ in lh_specs if k in lh_runs_t]
            min_l_vx_r = min(round(r["official_benchmarks"]["voxceleb1_o"]["eer_percent"], 2) for r in present_lh)
            min_l_vx_a = min(round(r["official_benchmarks"]["voxceleb1_o"]["as_norm"]["eer_percent"], 2) for r in present_lh)
            min_l_vx_q = min(round(r["official_benchmarks"]["voxceleb1_o"]["qmf"]["eer_percent"], 2) for r in present_lh)
            min_l_cn_q = min(round(r["official_benchmarks"]["cnceleb_e"]["qmf"]["eer_percent"], 2) for r in present_lh)
            min_l_cn_m = min(round(r["official_benchmarks"]["cnceleb_e"]["multi_enroll_map"]["as_norm"]["eer_percent"], 2) for r in present_lh)
            min_l_ls_q = min(round(r["official_benchmarks"]["librispeech_test_clean"]["qmf"]["eer_percent"], 2) for r in present_lh)
            min_l_ls_3 = min(round(r["official_benchmarks"]["librispeech_test_clean"]["multi_enroll_3utt"]["as_norm"]["eer_percent"], 2) for r in present_lh)
            min_l_p1 = min(round(d1(r, "universal_pooled")["eer_percent"], 2) for r in present_lh)
            min_l_p3 = min(round(d3(r, "universal_pooled")["eer_percent"], 2) for r in present_lh)
            for lkey, llbl in lh_specs:
                if lkey not in lh_runs_t:
                    continue
                ld = lh_runs_t[lkey]
                lob = ld["official_benchmarks"]
                lvx, lcn, lls = lob["voxceleb1_o"], lob["cnceleb_e"], lob["librispeech_test_clean"]
                lcn_me = lcn["multi_enroll_map"]["as_norm"]["eer_percent"]
                lls_3 = lls["multi_enroll_3utt"]["as_norm"]["eer_percent"]
                lp1 = d1(ld, "universal_pooled")["eer_percent"]
                lp3 = d3(ld, "universal_pooled")["eer_percent"]
                vr_s = f"\\textbf{{{f2(lvx['eer_percent'])}}}" if round(lvx["eer_percent"], 2) == min_l_vx_r else f2(lvx["eer_percent"])
                va_s = f"\\textbf{{{f2(lvx['as_norm']['eer_percent'])}}}" if round(lvx["as_norm"]["eer_percent"], 2) == min_l_vx_a else f2(lvx["as_norm"]["eer_percent"])
                vq_s = f"\\textbf{{{f2(lvx['qmf']['eer_percent'])}}}" if round(lvx["qmf"]["eer_percent"], 2) == min_l_vx_q else f2(lvx["qmf"]["eer_percent"])
                cq_s = f"\\textbf{{{f2(lcn['qmf']['eer_percent'])}}}" if round(lcn["qmf"]["eer_percent"], 2) == min_l_cn_q else f2(lcn["qmf"]["eer_percent"])
                cm_s = f"\\textbf{{{f2(lcn_me)}}}" if round(lcn_me, 2) == min_l_cn_m else f2(lcn_me)
                lq_s = f"\\textbf{{{f2(lls['qmf']['eer_percent'])}}}" if round(lls["qmf"]["eer_percent"], 2) == min_l_ls_q else f2(lls["qmf"]["eer_percent"])
                l3_s = f"\\textbf{{{f2(lls_3)}}}" if round(lls_3, 2) == min_l_ls_3 else f2(lls_3)
                p1_s = f"\\textbf{{{f2(lp1)}}}" if round(lp1, 2) == min_l_p1 else f2(lp1)
                p3_s = f"\\textbf{{{f2(lp3)}}}" if round(lp3, 2) == min_l_p3 else f2(lp3)
                vlines.append(
                    f"{llbl} & {int(ld['num_parameters'])/1e6:.2f} & {int(ld['num_steps']):,} & "
                    f"{vr_s}/{va_s}/{vq_s} & "
                    f"{f2(lcn['eer_percent'])}/{cq_s}/{cm_s} & "
                    f"{f2(lls['eer_percent'])}/{lq_s}/{l3_s} & "
                    f"{p1_s}/{p3_s} \\\\"
                )
        vtbl = (
            "\\begin{tabular}{@{}lrrcccc@{}}\n\\toprule\n"
            "\\textbf{\\texttt{v0.7.0} Arm (113 sh.) / Horizon Run (145 sh.)} & \\textbf{M par.} & \\textbf{Steps} & "
            "\\textbf{Vox1-O} & \\textbf{CN-E} & "
            "\\textbf{LS-clean} & \\textbf{Pooled 1/3} \\\\\n"
            " & & & (Raw/ASN/QMF) & (Raw/QMF/Multi) & (Raw/QMF/3u) & (Raw) \\\\\n\\midrule\n"
            + "\n".join(vlines) + "\n\\bottomrule\n\\end{tabular}\n"
        )
        emit(os.path.join(out_dir, "table_v070_ablation.tex"), vtbl)

    # ------------------------------------------------ Table: 16-config PFAS sweep
    plines = []
    cands = pfas_sweep["candidates"]
    for item in pfas_sweep["ranking"]:
        cname = item["name"]
        c = cands[cname]
        sp = c["spec"]
        v1, v3 = c["val_1utt_metrics"], c["val_3utt_metrics"]
        anchor = f"Yes ({sp['local_value_scale']:.2f})" if sp["use_global_anchor"] else "No"
        lstr = "GE2E" if sp["loss_type"] == "ge2e_softmax" else ("Pure ES" if sp["pure_extended_set"] else "BXS$^{*}$")
        comp_s = f"\\textbf{{{f2(c['composite_val_eer_percent'])}}}" if item["rank"] == 1 else f2(c["composite_val_eer_percent"])
        v1_s = f"\\textbf{{{f2(v1['eer_percent'])}}}" if item["rank"] == 1 else f2(v1["eer_percent"])
        v3_s = f"\\textbf{{{f2(v3['eer_percent'])}}}" if item["rank"] == 1 else f2(v3["eer_percent"])
        plines.append(
            f"{item['rank']} & \\texttt{{{tex_escape(cname)}}} & \\texttt{{{tex_escape(sp['attention_type'])}}} & "
            f"${sp['num_keys']}\\times({sp['key_dim']},{sp['value_dim']})$ & {c['representation_dim']} & "
            f"{anchor} & {lstr} & {v1_s} [{f2(v1['eer_ci95_low_percent'])}, {f2(v1['eer_ci95_high_percent'])}] & "
            f"{v3_s} [{f2(v3['eer_ci95_low_percent'])}, {f2(v3['eer_ci95_high_percent'])}] & {comp_s} \\\\"
        )
    ptbl = (
        "\\begin{tabular}{@{}rlllrllccc@{}}\n\\toprule\n"
        "\\textbf{Rk} & \\textbf{Candidate ID} & \\textbf{Attn.} & \\textbf{$K{\\times}(d_k,d_v)$} & "
        "\\textbf{Dim} & \\textbf{Anchor} & \\textbf{Loss} & \\textbf{Val 1-utt [95\\% CI]} & "
        "\\textbf{Val 3-utt [95\\% CI]} & \\textbf{Comp.} \\\\\n\\midrule\n"
        + "\n".join(plines) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    emit(os.path.join(out_dir, "table_pfas_sweep.tex"), ptbl)

    # ---------------------------------------------------- Table: all univ / single
    def zoo_table(mids: list[str], with_train: bool) -> str:
        rows = []
        for mid in mids:
            e = ex[mid]
            rows.append((
                mid,
                e["train_dataset"],
                int(e["num_steps"]),
                f"{e['backbone']} ({e['size_variant']})",
                f"{e['pooling_type']} / {LOSS_SHORT[e['loss_type']]} / {SCORE_SHORT[e['scoring_type']]}",
                int(e["num_parameters"]),
                d1(e, "librispeech")["eer_percent"],
                d3(e, "librispeech")["eer_percent"],
                d1(e, "cnceleb2")["eer_percent"],
                d3(e, "cnceleb2")["eer_percent"],
                d1(e, "voxceleb1")["eer_percent"],
                d3(e, "voxceleb1")["eer_percent"],
                macro(e, 1)["eer_percent"],
                macro(e, 3)["eer_percent"],
                d1(e, "universal_pooled")["eer_percent"],
                d3(e, "universal_pooled")["eer_percent"],
            ))
        cols = list(zip(*rows))
        bolded = {i: bold_min(list(cols[i])) for i in range(6, 16)}
        lines = []
        prev_ds = None
        for r_i, r in enumerate(rows):
            if with_train and prev_ds is not None and r[1] != prev_ds:
                lines.append("\\midrule")
            prev_ds = r[1]
            name = f"\\href{{https://huggingface.co/wq2012/{r[0]}}}{{\\texttt{{{tex_escape(r[0])}}}}}"
            if r[0] in TFLITE_REPOS:
                name += "$^{\\dagger}$"
            cells = [name]
            if with_train:
                cells.append(f"\\texttt{{{r[1]}}} ({r[2]:,})")
            else:
                cells.append(f"{r[2]:,}")
            cells += [f"\\texttt{{{tex_escape(r[3])}}}", f"{r[4]}", f"{r[5]:,}"]
            for i in range(6, 16, 2):
                cells.append(f"{bolded[i][r_i]} / {bolded[i + 1][r_i]}")
            lines.append(" & ".join(cells) + " \\\\")
        step_col = "\\textbf{Train (steps)} & " if with_train else "\\textbf{Steps} & "
        head = (
            "\\textbf{Hugging Face repo (\\texttt{wq2012/\\ldots})} & "
            + step_col
            + "\\textbf{Backbone (size)} & \\textbf{Pool / objective / scorer} & \\textbf{Params} & "
            "\\textbf{LS 1/3-utt} & \\textbf{CN 1/3-utt} & \\textbf{VOX 1/3-utt} & "
            "\\textbf{Macro 1/3-utt} & \\textbf{Pooled 1/3-utt} \\\\"
        )
        spec = "@{}l" + ("l" if with_train else "r") + "llrccccc@{}"
        return f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{head}\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"

    emit(os.path.join(out_dir, "table_univ_all.tex"), zoo_table(univ_order, with_train=False))
    emit(os.path.join(out_dir, "table_single_all.tex"), zoo_table(single_order, with_train=True))

    # ------------------------------- Table: minDCF / AUC / Vox1-O subset for all 48
    rows = []
    all_48_order = FLAGSHIP_ORDER + univ_order + single_order
    for mid in all_48_order:
        e = ex[mid]
        v = e["cross_dataset_eval"]["vox1_o_10spk_subset"]
        rows.append((
            mid,
            macro(e, 1)["min_dcf_001"],
            macro(e, 3)["min_dcf_001"],
            d1(e, "universal_pooled")["roc_auc"],
            d3(e, "universal_pooled")["roc_auc"],
            v["eer_percent"],
            v["eer_ci95_low_percent"],
            v["eer_ci95_high_percent"],
            v["min_dcf_001"],
        ))
    cols = list(zip(*rows))
    b1 = bold_min(list(cols[1]), f4)
    b2 = bold_min(list(cols[2]), f4)
    a1 = bold_max(list(cols[3]), f4)
    a2 = bold_max(list(cols[4]), f4)
    bv = bold_min(list(cols[5]))
    bvd = bold_min(list(cols[8]), f4)
    lines = []
    for i, r in enumerate(rows):
        if i in (len(FLAGSHIP_ORDER), len(FLAGSHIP_ORDER) + len(univ_order)):
            lines.append("\\midrule")
        lines.append(
            f"\\texttt{{{tex_escape(r[0])}}} & {b1[i]} / {b2[i]} & {a1[i]} / {a2[i]} & "
            f"{bv[i]} [{f2(r[6])}, {f2(r[7])}] & {bvd[i]} \\\\"
        )
    tbl = (
        "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
        "\\textbf{Model} & \\textbf{Macro minDCF 1/3} & \\textbf{Pooled AUC 1/3} & "
        "\\textbf{Vox1-O subset EER [95\\% CI]} & \\textbf{Subset minDCF} \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    emit(os.path.join(out_dir, "table_dcf_auc_vox.tex"), tbl)

    # --------------------------------------------------------------- Table: ablations
    def row_from_md(label: str, mde: dict, copied: bool = False) -> str:
        d_1, d_3 = mde["domains_1utt"], mde["domains_3utt"]
        p1 = d_1["universal_pooled"]
        cells = [label + ("$^{\\S}$" if copied else "")]
        for dom in ("librispeech", "cnceleb2", "voxceleb1"):
            cells.append(f"{f2(d_1[dom]['eer_percent'])} / {f2(d_3[dom]['eer_percent'])}")
        cells.append(f"{f2(mde['universal_macro_avg_1utt']['eer_percent'])} / {f2(mde['universal_macro_avg_3utt']['eer_percent'])}")
        cells.append(f"{f2(p1['eer_percent'])} [{f2(p1['eer_ci95_low_percent'])}, {f2(p1['eer_ci95_high_percent'])}] / {f2(d_3['universal_pooled']['eer_percent'])}")
        return " & ".join(cells) + " \\\\"

    lines = ["\\multirow{3}{*}{\\textbf{1. Stratified batching}}"]
    lines.append(" & " + row_from_md("$p_{\\text{within}} = 0.00$ (pure cross-domain, 4,600 steps)", dm["p_within_0.0"]["multi_domain_eval"]))
    lines.append(" & " + row_from_md("$p_{\\text{within}} = 0.75$ (default hybrid, 4,600 steps)", dm["p_within_0.75"]["multi_domain_eval"], copied=True))
    lines.append(" & " + row_from_md("$p_{\\text{within}} = 1.00$ (pure within-domain, 4,600 steps)", dm["p_within_1.0"]["multi_domain_eval"]))
    lines.append("\\midrule")

    s2_rows = [
        ("Pure ext.-set softmax (paper Eq.~2), cosine", xs["ext_softmax_pure_eq3"]["multi_domain_eval"], False),
        ("Blended ext.-set softmax (BXS$^{*}$), cosine", xs["ext_softmax_blended"]["multi_domain_eval"], True),
        ("3-term Dr-Vec$^{*}$ + pure ext.-set (Eq.~2)", xs["dr_vectors_3term_pure_eq7"]["multi_domain_eval"], False),
        ("1-term down-scaled Dr-Vec$^{*}$ + BXS$^{*}$", xs["dr_vectors_1term_blended"]["multi_domain_eval"], True),
    ]
    if "dr_vector_terms_and_contrast_ablation" in st:
        dvc = st["dr_vector_terms_and_contrast_ablation"]
        s2_rows.append(("3-term Dr-Vec$^{*}$ sym.\\ skip + BXS$^{*}$", dvc["dr_vector_3term_sym_skip"]["multi_domain_eval"], False))
        s2_rows.append(("Hybrid \\texttt{ge2e\\_contrast} surrogate$^{*}$", dvc["ge2e_contrast_stabilized"]["multi_domain_eval"], True))
    if "pfas_sn_attention_and_anchor_ablation" in st:
        pfa = st["pfas_sn_attention_and_anchor_ablation"]
        s2_rows.append(("PFAS-SN$^{*}$ \\texttt{shared\\_nonlinear} + ATP anchor", pfa["pfas_sn_mlp_with_anchor"]["multi_domain_eval"], True))
        s2_rows.append(("PFAS-SN$^{*}$ \\texttt{shared\\_nonlinear}, no anchor", pfa["pfas_sn_mlp_no_anchor"]["multi_domain_eval"], False))
        s2_rows.append(("PFAS-SN$^{*}$ \\texttt{dot} attention + ATP anchor", pfa["pfas_sn_dot_with_anchor"]["multi_domain_eval"], False))
        s2_rows.append(("PFAS-SN$^{*}$ \\texttt{maxpool} + ATP anchor", pfa["pfas_sn_maxpool_with_anchor"]["multi_domain_eval"], False))
    lines.append(f"\\multirow{{{len(s2_rows)}}}{{*}}{{\\textbf{{2. Objectives \\& heads}}}}")
    for lbl, mde, cp in s2_rows:
        lines.append(" & " + row_from_md(lbl, mde, copied=cp))
    lines.append("\\midrule")

    lines.append("\\multirow{4}{*}{\\textbf{3. Training crops}}")
    lines.append(" & " + row_from_md("$T \\sim \\mathcal{U}(90,180)$, full-seq ($\\le 3{,}000$f) eval", vl["variable_length_true"]["full_seq_eval"], copied=True))
    lines.append(" & " + row_from_md("Fixed $T{=}180$, full-seq ($\\le 3{,}000$f) eval", vl["variable_length_false"]["full_seq_eval"]))
    lines.append(" & " + row_from_md("$T \\sim \\mathcal{U}(90,180)$, short 1.0\\,s (100f) eval", vl["variable_length_true"]["short_1s_eval"], copied=True))
    lines.append(" & " + row_from_md("Fixed $T{=}180$, short 1.0\\,s (100f) eval", vl["variable_length_false"]["short_1s_eval"]))
    lines.append("\\midrule")

    h_rows = []
    for k, lbl, cp in (
        ("steps_400", "400 steps (own 400-step cosine schedule)", False),
        ("steps_800", "800 steps (own 800-step cosine schedule)", False),
        ("steps_1200", "1,200 steps (own 1,200-step cosine schedule)", False),
        ("steps_2400", "2,400 steps (own 2,400-step cosine schedule)", False),
        ("steps_4600", "4,600 steps ($S_{\\text{safe}}$, own 4,600-step schedule)", True),
    ):
        if k in own_h:
            h_rows.append((lbl, own_h[k]["multi_domain_eval"], cp))
    lines.append(f"\\multirow{{{len(h_rows)}}}{{*}}{{\\textbf{{4. Horizon (own LR)}}}}")
    for lbl, mde, cp in h_rows:
        lines.append(" & " + row_from_md(lbl, mde, copied=cp))
    lines.append("\\midrule")

    s5_rows = [
        ("Mamba unidirectional on LibriSpeech (2,400 steps)", m_ls["unidirectional"]["multi_domain_eval"], True),
        ("Mamba bidirectional on LibriSpeech (2,400 steps)", m_ls["bidirectional"]["multi_domain_eval"], False),
    ]
    if "mamba_direction_ablation" in st:
        m_u = st["mamba_direction_ablation"]
        s5_rows.append(("Mamba unidirectional on Universal (4,600 steps)", m_u["mamba_unidirectional"]["multi_domain_eval"], True))
        s5_rows.append(("Mamba bidirectional on Universal (4,600 steps)", m_u["mamba_bidirectional"]["multi_domain_eval"], False))
    lines.append(f"\\multirow{{{len(s5_rows)}}}{{*}}{{\\textbf{{5. Mamba direction}}}}")
    for lbl, mde, cp in s5_rows:
        lines.append(" & " + row_from_md(lbl, mde, copied=cp))
    lines.append("\\midrule")

    s6_rows = []
    if "ecapa_norm_order_ablation" in st:
        ec = st["ecapa_norm_order_ablation"]
        s6_rows.append(("ECAPA-TDNN BatchNorm after activation (v0.5.0 default)", ec["ecapa_bn_style_after_act"]["multi_domain_eval"], True))
        s6_rows.append(("ECAPA-TDNN LayerNorm before activation (legacy v0.4.0)", ec["ecapa_ln_before_act"]["multi_domain_eval"], False))
    if "vad_and_agc_ablation" in st:
        va = st["vad_and_agc_ablation"]
        s6_rows.append(("Frontend VAD 26\\,dB + peak AGC (MUSAN+RIR shards)", va["vad_26db_with_agc"]["multi_domain_eval"], True))
        s6_rows.append(("Frontend VAD 26\\,dB, no peak AGC (\\texttt{FeatureCache})$^{\\sharp}$", va["vad_26db_no_agc"]["multi_domain_eval"], False))
        s6_rows.append(("Frontend VAD 20\\,dB, no peak AGC (\\texttt{FeatureCache})$^{\\sharp}$", va["vad_20db_no_agc"]["multi_domain_eval"], False))
    lines.append(f"\\multirow{{{len(s6_rows)}}}{{*}}{{\\textbf{{6. Norm \\& VAD/AGC}}}}")
    for lbl, mde, cp in s6_rows:
        lines.append(" & " + row_from_md(lbl, mde, copied=cp))

    if "paper_faithful_presets_smoke_test" in st:
        pf = st["paper_faithful_presets_smoke_test"]
        lines.append("\\midrule")
        lines.append("\\multirow{4}{*}{\\textbf{7. Paper presets}}")
        lines.append(" & " + row_from_md("\\texttt{icassp2018\\_ge2e} ($3{\\times}768$ LSTM + 256 proj)", pf["icassp2018_ge2e_pure_paper"]["multi_domain_eval"]))
        lines.append(" & " + row_from_md("\\texttt{interspeech2021\\_dr\\_vectors} ($200{+}56$, $3{\\times}256$ LReLU)", pf["interspeech2021_dr_vectors_pure_paper"]["multi_domain_eval"]))
        lines.append(" & " + row_from_md("\\texttt{odyssey2022\\_conformer\\_atp} (conv-1st, sin PE, $\\times 2$ sub)", pf["odyssey2022_conformer_atp_pure_paper"]["multi_domain_eval"]))
        lines.append(" & " + row_from_md("\\texttt{odyssey2022\\_pfas} (\\texttt{atp\\_ffn}, $32{\\times}(16{+}48)$)", pf["odyssey2022_pfas_pure_paper"]["multi_domain_eval"]))

    tbl = (
        "\\begin{tabular}{@{}l>{\\raggedright\\arraybackslash}p{6.5cm}ccccc@{}}\n\\toprule\n"
        "\\textbf{Study} & \\textbf{Variant} & \\textbf{LS 1/3-utt} & \\textbf{CN 1/3-utt} & "
        "\\textbf{VOX 1/3-utt} & \\textbf{Macro 1/3-utt} & \\textbf{Pooled 1-utt [95\\% CI] / 3-utt} \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    emit(os.path.join(out_dir, "table_ablations.tex"), tbl)

    # ------------------------------------------------------------- Table: TFLite
    tfl_lines = []
    for fmid in (
        "univ_full_conformer_large_arcface",
        "univ_full_ecapa_c1024_arcface",
        "univ_full_resnet34_arcface",
        "univ_full_eres2conformer_large",
        "univ_full_eres2conformer_v2_large",
        "univ_full_eres2conformer_xl",
        "univ_full_eres2conformer_v2_diar",
        "univ_full_eres2conformer_xl_diar",
        "univ_full_eres2conformer_xl_diar_adapt",
    ):
        te = ex.get(fmid, {}).get("tflite_eval", {})
        if te:
            cos_min_s = f"{te['cos_fp32_int8_min']:.6f}" if "cos_fp32_int8_min" in te else "---"
            fp_ci_s = (
                f"{f2(te['eer_percent'])} [{f2(te['eer_ci95_fp32'][0])}, {f2(te['eer_ci95_fp32'][1])}]"
                if "eer_ci95_fp32" in te
                else f2(te["eer_percent"])
            )
            int_ci_s = (
                f"{f2(te['eer_percent_int8'])} [{f2(te['eer_ci95_int8'][0])}, {f2(te['eer_ci95_int8'][1])}]"
                if "eer_ci95_int8" in te
                else f2(te["eer_percent_int8"])
            )
            dcf_s = (
                f"{f4(te['min_dcf_001'])} / {f4(te['min_dcf_001_int8'])}"
                if "min_dcf_001" in te and "min_dcf_001_int8" in te
                else "---"
            )
            tfl_lines.append(
                f"\\texttt{{{tex_escape(fmid)}}} & \\texttt{{voxceleb1\\_o}} & {f2(te['jax_full_seq_eer_percent'])} & "
                f"{fp_ci_s} & {int_ci_s} & {te['eer_percent_int8'] - te['eer_percent']:+.2f} & "
                f"{dcf_s} & {te['cos_fp32_int8_mean']:.6f} / {cos_min_s} & "
                f"{f2(te['latency_ms_fp32_mean_50runs'])} / {f2(te['latency_ms_int8_mean_50runs'])} \\\\"
            )
    if tfl_lines:
        tfl_lines.append("\\midrule")
    for ttag, tmid in tfl_tags:
        te = ex[tmid].get("tflite_eval", {})
        if not te:
            continue
        prim = te["primary_domain"]
        rec32 = te["domains_1utt_fp32"][prim]
        rec8 = te["domains_1utt_int8"][prim]

        def eer_ci(rec: dict) -> str:
            if "eer_ci95_low" in rec:
                return f"{f2(rec['eer_percent'])} [{f2(rec['eer_ci95_low'])}, {f2(rec['eer_ci95_high'])}]"
            return f2(rec["eer_percent"])

        tfl_lines.append(
            f"\\texttt{{{tex_escape(tmid)}}} & \\texttt{{{tex_escape(prim)}}} & {f2(te['jax_full_seq_eer_percent'])} & "
            f"{eer_ci(rec32)} & {eer_ci(rec8)} & {te['eer_percent_int8'] - te['eer_percent']:+.2f} & "
            f"{f4(rec32['min_dcf_001'])} / {f4(rec8['min_dcf_001'])} & "
            f"{te['cos_fp32_int8_mean']:.6f} / {te['cos_fp32_int8_min']:.6f} & "
            f"{f2(te['latency_ms_fp32_mean_50runs'])} / {f2(te['latency_ms_int8_mean_50runs'])} \\\\"
        )
    has_ci = any(
        "eer_ci95_low" in ex[tmid]["tflite_eval"]["domains_1utt_fp32"][ex[tmid]["tflite_eval"]["primary_domain"]]
        for _, tmid in tfl_tags
        if ex[tmid].get("tflite_eval")
    )
    ci_hdr = " [95\\% CI]" if has_ci else ""
    tbl = (
        "\\begin{tabular}{@{}llccccccc@{}}\n\\toprule\n"
        "\\textbf{Model} & \\textbf{Eval set} & \\textbf{JAX EER} & "
        f"\\textbf{{FP32 EER{ci_hdr}}} & \\textbf{{INT8 EER{ci_hdr}}} & \\textbf{{$\\Delta$}} & "
        "\\textbf{minDCF FP32/INT8} & \\textbf{cos mean/min} & \\textbf{Lat.\\ FP32/INT8 (ms)} \\\\\n\\midrule\n"
        + "\n".join(tfl_lines) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    emit(os.path.join(out_dir, "table_tflite.tex"), tbl)

    # ------------------------------------------- Table: universal vs. specialist deltas
    dl = []
    for m_id, t_id in pairs:
        pu, pl = d1(ex[m_id], "universal_pooled")["eer_percent"], d1(ex[t_id], "universal_pooled")["eer_percent"]
        mu, ml = macro(ex[m_id], 1)["eer_percent"], macro(ex[t_id], 1)["eer_percent"]
        p3u, p3l = d3(ex[m_id], "universal_pooled")["eer_percent"], d3(ex[t_id], "universal_pooled")["eer_percent"]
        dl.append((pair_label(m_id), pl, pu, pu - pl, mu - ml, p3u - p3l))
    dl.sort(key=lambda r: r[3])
    dlines = [f"{r[0]} & {f2(r[1])} & {f2(r[2])} & {r[3]:+.2f} & {r[4]:+.2f} & {r[5]:+.2f} \\\\" for r in dl]
    tbl = (
        "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
        "\\textbf{Backbone / objective} & \\textbf{\\texttt{ls\\_*} pooled 1-utt} & \\textbf{\\texttt{univ\\_*} pooled 1-utt} & "
        "\\textbf{$\\Delta$ pooled 1-utt} & \\textbf{$\\Delta$ macro 1-utt} & \\textbf{$\\Delta$ pooled 3-utt} \\\\\n\\midrule\n"
        + "\n".join(dlines) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    emit(os.path.join(out_dir, "table_univ_vs_ls.tex"), tbl)

    # ------------------------------------------------------------ Table: seed floor
    seed_lines = []
    for mid in ("univ_conformer_small_ge2e", "univ_ecapa_tdnn_small_ge2e", "ls_conformer_small_ge2e"):
        if mid not in ms_all:
            continue
        mv = ms_all[mid]
        s1, s3 = mv["summary_1utt"], mv["summary_3utt"]
        per = sorted(mv["per_seed"], key=lambda r: int(r["seed"]))
        seeds_str = ", ".join(str(r["seed"]) for r in per)
        vals = "/".join(f2(r["summary"]["pooled_1utt_eer"]) for r in per)
        main_e = ex[mid]
        seed_lines.append(
            f"\\texttt{{{tex_escape(mid)}}} ({int(main_e['num_steps']):,}) & {seeds_str} & {vals} & "
            f"{f2(s1['universal_pooled']['mean'])}$\\pm${f2(s1['universal_pooled']['std'])} & "
            f"{f2(s1['macro_avg']['mean'])}$\\pm${f2(s1['macro_avg']['std'])} & "
            f"{f2(s1['librispeech']['mean'])}$\\pm${f2(s1['librispeech']['std'])} & "
            f"{f2(s3['universal_pooled']['mean'])}$\\pm${f2(s3['universal_pooled']['std'])} & "
            f"{f2(d1(main_e, 'universal_pooled')['eer_percent'])}/{f2(macro(main_e, 1)['eer_percent'])} \\\\"
        )
    tbl = (
        "\\begin{tabular}{@{}llcccccc@{}}\n\\toprule\n"
        "\\textbf{Model (steps)} & \\textbf{Seeds} & \\textbf{Pooled 1-utt} & \\textbf{Pooled 1-utt} & "
        "\\textbf{Macro 1-utt} & \\textbf{LS 1-utt} & \\textbf{Pooled 3-utt} & "
        "\\textbf{Released (42)} \\\\\n"
        " & & (per seed) & \\multicolumn{4}{c}{(mean $\\pm$ sd over seeds)} & (pooled/macro) \\\\\n\\midrule\n"
        + "\n".join(seed_lines) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    emit(os.path.join(out_dir, "table_seed_floor.tex"), tbl)

    # ---------------------------------------------------- Table: matched-data control
    def mrow(label: str, steps: int, spk: int, utt: int, mde: dict) -> str:
        p1 = mde["domains_1utt"]["universal_pooled"]
        return (
            f"{label} & {steps:,} & {spk}/{utt:,} & "
            f"{f2(mde['domains_1utt']['librispeech']['eer_percent'])} & "
            f"{f2(mde['domains_1utt']['cnceleb2']['eer_percent'])} & "
            f"{f2(mde['domains_1utt']['voxceleb1']['eer_percent'])} & "
            f"{f2(mde['universal_macro_avg_1utt']['eer_percent'])} & "
            f"{f2(p1['eer_percent'])} [{f2(p1['eer_ci95_low_percent'])}, {f2(p1['eer_ci95_high_percent'])}] & "
            f"{f2(mde['universal_macro_avg_3utt']['eer_percent'])} & "
            f"{f2(mde['domains_3utt']['universal_pooled']['eer_percent'])} \\\\"
        )

    mlines = [
        mrow("\\texttt{ls\\_conformer\\_small\\_ge2e} (shards)$^{\\S}$", 2400, 232, 5788, ex["ls_conformer_small_ge2e"]["multi_domain_eval"]),
    ]
    if "univ_subsampled_5788_2400steps" in md_ctrl:
        m24 = md_ctrl["univ_subsampled_5788_2400steps"]
        mlines.append(mrow("Univ.\\ 5,788-utt (\\texttt{FeatureCache}, matched)$^{\\sharp}$", 2400, m24["num_train_speakers"], m24["num_train_utterances"], m24["multi_domain_eval"]))
    if "univ_subsampled_5788_4600steps" in md_ctrl:
        m46 = md_ctrl["univ_subsampled_5788_4600steps"]
        mlines.append(mrow("Univ.\\ 5,788-utt (\\texttt{FeatureCache}, $S_{\\text{safe}}$)$^{\\sharp}$", 4600, m46["num_train_speakers"], m46["num_train_utterances"], m46["multi_domain_eval"]))
    mlines.append(mrow("\\texttt{univ\\_conformer\\_small\\_ge2e} (shards)$^{\\S}$", 4600, 381, 9447, ex["univ_conformer_small_ge2e"]["multi_domain_eval"]))
    tbl = (
        "\\begin{tabular}{@{}lrcccccccc@{}}\n\\toprule\n"
        "\\textbf{Training configuration} & \\textbf{Steps} & \\textbf{Spk/utts} & "
        "\\textbf{LS} & \\textbf{CN} & \\textbf{VOX} & \\textbf{Macro 1} & "
        "\\textbf{Pooled 1 [95\\% CI]} & \\textbf{Macro 3} & \\textbf{Pooled 3} \\\\\n\\midrule\n"
        + "\n".join(mlines) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    emit(os.path.join(out_dir, "table_matched_data.tex"), tbl)

    # ------------------------------------ Table: Diarization SV (main + appendix)
    if v080_abl is not None:
        zs_m = v080_abl.get("v070_zero_shot_models", {})
        rel_m = v080_abl.get("v080_released_flagships", {})
        fus_m = v080_abl.get("v080_7corpus_fusion", {})
        abl_m = v080_abl.get("v080_controlled_ablations", {})

        d_entries: list[tuple[str, str, float, dict, dict, bool]] = []
        for zlbl, zmid in (
            ("ECAPA-c1024 (ZS)", "univ_full_ecapa_c1024_arcface"),
            ("ResNet-34 (ZS)", "univ_full_resnet34_arcface"),
            ("ERes2Conf-L (ZS)", "univ_full_eres2conformer_large"),
            ("ERes2Conf-V2 (ZS)", "univ_full_eres2conformer_v2_large"),
            ("ERes2Conf-V2-PFAS (ZS)", "univ_full_eres2conformer_v2_pfas"),
            ("ERes2Conf-XL-PFAS (ZS)", "univ_full_eres2conformer_xl_pfas"),
            ("ERes2Conf-XL (ZS)", "univ_full_eres2conformer_xl"),
        ):
            if zmid in zs_m and zmid in fm:
                d_entries.append((
                    f"\\href{{https://huggingface.co/wq2012/{zmid}}}{{{zlbl}}}",
                    "3 (12.2k)",
                    fm[zmid]["official_benchmarks"]["voxceleb1_o"]["qmf"]["eer_percent"],
                    zs_m[zmid]["diarization_benchmarks"]["suites"],
                    zs_m[zmid]["diarization_benchmarks"]["macro_averages"],
                    False,
                ))
        for rlbl, rmid in (
            ("eres2conformer\\_v2\\_diar$^{\\dagger}$", "univ_full_eres2conformer_v2_diar"),
            ("eres2conformer\\_xl\\_diar$^{\\dagger}$", "univ_full_eres2conformer_xl_diar"),
            ("eres2conformer\\_xl\\_diar\\_pfas", "univ_full_eres2conformer_xl_diar_pfas"),
            ("eres2conformer\\_xl\\_diar\\_adapt$^{\\dagger}$", "univ_full_eres2conformer_xl_diar_adapt"),
        ):
            if rmid in rel_m:
                d_entries.append((
                    f"\\href{{https://huggingface.co/wq2012/{rmid}}}{{\\texttt{{{rlbl}}}}}",
                    "7 (24.1k)",
                    rel_m[rmid]["official_benchmarks"]["voxceleb1_o"]["qmf"]["eer_percent"],
                    rel_m[rmid]["diarization_benchmarks"]["suites"],
                    rel_m[rmid]["diarization_benchmarks"]["macro_averages"],
                    True,
                ))
        if fus_m and "diarization_benchmarks" in fus_m:
            nf_v8 = len(fus_m.get("models", []))
            d_entries.append((
                f"{nf_v8}-Way 7-Corpus Score Fusion",
                "7 (24.1k)",
                fus_m["official_benchmarks"]["voxceleb1_o"]["qmf"]["eer_percent"],
                fus_m["diarization_benchmarks"]["suites"],
                fus_m["diarization_benchmarks"]["macro_averages"],
                True,
            ))

        if d_entries:
            min_dvx = min(round(e[2], 2) for e in d_entries)
            min_fws = min(round(e[3]["fisher_test_within_session"]["qmf"]["eer_percent"], 2) for e in d_entries)
            min_cws = min(round(e[3]["callhome_test_within_session"]["qmf"]["eer_percent"], 2) for e in d_entries)
            min_aws = min(round(e[3]["ami_test_within_session"]["qmf"]["eer_percent"], 2) for e in d_entries)
            min_iws = min(round(e[3]["icsi_test_within_session"]["qmf"]["eer_percent"], 2) for e in d_entries)
            min_mws = min(round(e[4]["macro_4corpus_within_session"]["qmf"]["eer_percent"], 2) for e in d_entries)
            min_mst = min(round(e[4]["macro_4corpus_standard"]["qmf"]["eer_percent"], 2) for e in d_entries)
            min_msh = min(round(e[4]["macro_4corpus_short_turn"]["qmf"]["eer_percent"], 2) for e in d_entries)

            dlines_tex = []
            prev_is_v8 = False
            for idx_e, (elbl, ecorp, evx, esd, emd, is_v8) in enumerate(d_entries):
                if idx_e > 0 and is_v8 and not prev_is_v8:
                    dlines_tex.append("\\midrule")
                if elbl.endswith("Score Fusion"):
                    dlines_tex.append("\\midrule")
                prev_is_v8 = is_v8

                def _bcell(v: float, mv: float) -> str:
                    return f"\\textbf{{{f2(v)}}}" if round(v, 2) == mv else f2(v)

                def _corp_cell(cn: str, mws_c: float, sd_c: dict = esd) -> str:
                    w_v = sd_c[f"{cn}_test_within_session"]["qmf"]["eer_percent"]
                    s_v = sd_c[f"{cn}_test_standard"]["qmf"]["eer_percent"]
                    h_v = sd_c[f"{cn}_test_short_turn"]["qmf"]["eer_percent"]
                    return f"{_bcell(w_v, mws_c)}/{f2(s_v)}/{f2(h_v)}"

                m_ws = emd["macro_4corpus_within_session"]["qmf"]["eer_percent"]
                m_st = emd["macro_4corpus_standard"]["qmf"]["eer_percent"]
                m_sh = emd["macro_4corpus_short_turn"]["qmf"]["eer_percent"]
                dlines_tex.append(
                    f"{elbl} & {ecorp} & {_bcell(evx, min_dvx)} & "
                    f"{_corp_cell('fisher', min_fws)} & {_corp_cell('callhome', min_cws)} & "
                    f"{_corp_cell('ami', min_aws)} & {_corp_cell('icsi', min_iws)} & "
                    f"{_bcell(m_ws, min_mws)}/{_bcell(m_st, min_mst)}/{_bcell(m_sh, min_msh)} \\\\"
                )
            dtbl = (
                "\\begin{tabular}{@{}lccccccc@{}}\n\\toprule\n"
                "\\textbf{Model / Regime} & \\textbf{Corp.\\ (Spk)} & \\textbf{Vox1-O} & "
                "\\textbf{Fisher} & \\textbf{CallHome} & \\textbf{AMI} & \\textbf{ICSI} & "
                "\\textbf{4-Corpus Macro} \\\\\n"
                " & & (QMF) & (WS/Std/Sh) & (WS/Std/Sh) & (WS/Std/Sh) & (WS/Std/Sh) & (WS/Std/Sh QMF) \\\\\n\\midrule\n"
                + "\n".join(dlines_tex) + "\n\\bottomrule\n\\end{tabular}\n"
            )
            emit(os.path.join(out_dir, "table_diarization_sv.tex"), dtbl)

        if abl_m:
            da_specs: list[tuple[str, str, dict]] = []
            if "xl_diar_adapt_gentle_41k" in abl_m:
                da_specs.append(("1.\\ XL-Diar-Adapt (Gentle LR $0.012$, $41$k)", "Unfrozen SGD ($0.012$)", abl_m["xl_diar_adapt_gentle_41k"]))
            if "univ_full_eres2conformer_xl_diar_adapt" in rel_m:
                da_specs.append(("2.\\ XL-Diar-Adapt (LR $0.045$, $78$k)$^{\\star}$", "3k Align + SGD ($0.045$)", rel_m["univ_full_eres2conformer_xl_diar_adapt"]))
            if "univ_full_eres2conformer_xl_diar" in rel_m:
                da_specs.append(("3.\\ XL-Diar ($162$k From-Scratch)$^{\\star}$", "Scratch SGD ($0.10$)", rel_m["univ_full_eres2conformer_xl_diar"]))
            if "xl_diar_pfas_unfrozen_lr002" in abl_m:
                da_specs.append(("4.\\ XL-Diar-PFAS (Unfrozen $0.02\\times$, $20$k)", "Joint AdamW ($0.02\\times$)", abl_m["xl_diar_pfas_unfrozen_lr002"]))
            if "xl_diar_pfas_unfrozen_lr0004" in abl_m:
                da_specs.append(("5.\\ XL-Diar-PFAS (Unfrozen $0.004\\times$, $15$k)", "Joint AdamW ($0.004\\times$)", abl_m["xl_diar_pfas_unfrozen_lr0004"]))
            if "xl_diar_pfas_frozen_20k_v1" in abl_m:
                da_specs.append(("6.\\ XL-Diar-PFAS (Frozen on $41$k trunk, $20$k)", "100\\% Frozen Trunk", abl_m["xl_diar_pfas_frozen_20k_v1"]))
            if "univ_full_eres2conformer_xl_diar_pfas" in rel_m:
                da_specs.append(("7.\\ XL-Diar-PFAS (Frozen on $78$k trunk)$^{\\star}$", "100\\% Frozen Trunk", rel_m["univ_full_eres2conformer_xl_diar_pfas"]))
            if da_specs:
                dalines = []
                min_davx = min(round(x[2]["official_benchmarks"]["voxceleb1_o"]["qmf"]["eer_percent"], 2) for x in da_specs)
                min_daws = min(round(x[2]["diarization_benchmarks"]["macro_averages"]["macro_4corpus_within_session"]["qmf"]["eer_percent"], 2) for x in da_specs)
                min_dash = min(round(x[2]["diarization_benchmarks"]["macro_averages"]["macro_4corpus_short_turn"]["qmf"]["eer_percent"], 2) for x in da_specs)
                for dalbl, dareg, dad in da_specs:
                    vx_da = dad["official_benchmarks"]["voxceleb1_o"]
                    cn_da = dad["official_benchmarks"]["cnceleb_e"]["multi_enroll_map"].get(
                        "as_norm", dad["official_benchmarks"]["cnceleb_e"]["multi_enroll_map"]
                    )["eer_percent"]
                    sd_da = dad["diarization_benchmarks"]["suites"]
                    md_da = dad["diarization_benchmarks"]["macro_averages"]
                    vq_da = vx_da["qmf"]["eer_percent"]
                    mws_da = md_da["macro_4corpus_within_session"]["qmf"]["eer_percent"]
                    mst_da = md_da["macro_4corpus_standard"]["qmf"]["eer_percent"]
                    msh_da = md_da["macro_4corpus_short_turn"]["qmf"]["eer_percent"]
                    vq_s = f"\\textbf{{{f2(vq_da)}}}" if round(vq_da, 2) == min_davx else f2(vq_da)
                    mws_s = f"\\textbf{{{f2(mws_da)}}}" if round(mws_da, 2) == min_daws else f2(mws_da)
                    msh_s = f"\\textbf{{{f2(msh_da)}}}" if round(msh_da, 2) == min_dash else f2(msh_da)
                    dalines.append(
                        f"{dalbl} & {dareg} & {f2(vx_da['eer_percent'])}/{vq_s} & {f2(cn_da)} & "
                        f"{f2(sd_da['fisher_test_within_session']['eer_percent'])}/{f2(sd_da['fisher_test_within_session']['qmf']['eer_percent'])} & "
                        f"{f2(sd_da['ami_test_within_session']['eer_percent'])}/{f2(sd_da['ami_test_within_session']['qmf']['eer_percent'])} & "
                        f"{mws_s}/{f2(mst_da)}/{msh_s} \\\\"
                    )
                datbl = (
                    "\\begin{tabular}{@{}llccccc@{}}\n\\toprule\n"
                    "\\textbf{7-Corpus Adaptation / Hybrid-PFAS Arm} & \\textbf{Trunk Regime} & "
                    "\\textbf{Vox1-O} & \\textbf{CN-E} & \\textbf{Fisher WS} & \\textbf{AMI WS} & "
                    "\\textbf{4-Corpus Macro} \\\\\n"
                    " & & (Raw/QMF) & (Multi) & (Raw/QMF) & (Raw/QMF) & (WS/Std/Sh QMF) \\\\\n\\midrule\n"
                    + "\n".join(dalines) + "\n\\bottomrule\n\\end{tabular}\n"
                )
                emit(os.path.join(out_dir, "table_diarization_ablation.tex"), datbl)

    print(f"Wrote generated LaTeX to {out_dir}")
    print(f"  Spearman(pooled, Vox1-O subset) = {rho:.3f} (p={pval:.3e}); Spearman(in-VOX, Vox1-O) = {rho_v:.3f} (p={pval_v:.3e})")
    print(f"  univ wins pooled-1 in {len(pairs) - len(worse_pool1)}/{len(pairs)} pairs; macro-1 in {len(pairs) - len(worse_macro1)}/{len(pairs)} pairs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_dir", default=os.path.expanduser("~/Code/github/flax_speaker_paper"))
    args = parser.parse_args()
    main(args.paper_dir)
