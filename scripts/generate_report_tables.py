#!/usr/bin/env python3
"""Regenerate the number-bearing tables of RESEARCH_REPORT.md from the JSON archives.

Prints Markdown to stdout (used by the author when updating RESEARCH_REPORT.md), so
that no number in the report is typed by hand:

    python scripts/generate_report_tables.py > /tmp/report_tables.md
"""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PM = os.path.join(REPO_ROOT, "pretrained_models")


def load(name):
    with open(os.path.join(PM, name), "r", encoding="utf-8") as f:
        return json.load(f)


def f2(x):
    return f"{x:.2f}"


def main() -> None:
    bench = load("benchmark_results.json")
    abl = load("ablation_results.json")
    vox = load("vox1_o_10spk_subset_results.json")
    r2 = load("r2_seed_and_control_results.json")
    r3 = load("r3_fixed_crop_controls.json")
    fps = load("step0_fingerprints.json")
    ex = bench["experiments"]
    out = []

    # Vox1-O subset table
    out.append("### VoxCeleb1-O 10-speaker subset (all 44 models)\n")
    out.append(
        f"Protocol: `{vox['protocol']}` — {vox['num_trials']:,} trials "
        f"({vox['num_target_trials']:,} target / {vox['num_impostor_trials']:,} impostor, "
        f"{100 * vox['num_target_trials'] / vox['num_trials']:.1f}% target), {vox['num_unique_utterances']} utterances "
        f"from {len(vox['eval_speakers'])} held-out speakers (`{'`, `'.join(vox['eval_speakers'])}`), "
        f"VAD {vox['vad_energy_db_threshold']:.0f} dB (matched to training). "
        f"**Not comparable to published VoxCeleb1-O numbers.**\n"
    )
    out.append("| Model | Subset EER (%) [95% CI] | minDCF@0.01 | ROC-AUC | Pooled 1-utt EER (%) |")
    out.append("| :--- | :---: | :---: | :---: | :---: |")
    for mid, m in sorted(vox["models"].items(), key=lambda kv: kv[1]["eer_percent"]):
        pooled = ex[mid]["multi_domain_eval"]["domains_1utt"]["universal_pooled"]["eer_percent"]
        out.append(
            f"| `{mid}` | {f2(m['eer_percent'])} [{f2(m['eer_ci95_low_percent'])}, {f2(m['eer_ci95_high_percent'])}] | "
            f"{m['min_dcf_001']:.4f} | {m['roc_auc']:.4f} | {f2(pooled)} |"
        )
    out.append("")

    # Ablations
    def row(label, mde):
        d1, d3 = mde["domains_1utt"], mde["domains_3utt"]
        p1 = d1["universal_pooled"]
        return (
            f"| {label} | {f2(d1['librispeech']['eer_percent'])} / {f2(d3['librispeech']['eer_percent'])} | "
            f"{f2(d1['cnceleb2']['eer_percent'])} / {f2(d3['cnceleb2']['eer_percent'])} | "
            f"{f2(d1['voxceleb1']['eer_percent'])} / {f2(d3['voxceleb1']['eer_percent'])} | "
            f"{f2(mde['universal_macro_avg_1utt']['eer_percent'])} / {f2(mde['universal_macro_avg_3utt']['eer_percent'])} | "
            f"{f2(p1['eer_percent'])} [{f2(p1['eer_ci95_low_percent'])}, {f2(p1['eer_ci95_high_percent'])}] / "
            f"{f2(d3['universal_pooled']['eer_percent'])} |"
        )

    out.append("### Controlled ablations (`ablation_results.json`, `r2_seed_and_control_results.json`, `r3_fixed_crop_controls.json`; § = the released checkpoint, copied from the main benchmark — trained by an earlier code revision with **fixed 180-frame crops**)\n")
    out.append("| Variant | LS 1/3-utt | CN 1/3-utt | VOX 1/3-utt | Macro 1/3-utt | Pooled 1-utt [95% CI] / 3-utt |")
    out.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    dm = abl["domain_mixing_ablation"]
    vl = abl["variable_length_ablation"]
    fc = r3["domain_mixing_fixed_crop"]
    rerun = r2.get("seed42_rerun_current_code", {}).get("models", {})
    out.append(row("1. `within_domain_prob = 0.75`, fixed T = 180 (released checkpoint, earlier revision) §", dm["p_within_0.75"]["multi_domain_eval"]))
    out.append(row("1. `within_domain_prob = 0.0`, fixed T = 180 (`HEAD`, seed 42)", fc["p_within_0.0"]["multi_domain_eval"]))
    out.append(row("1. `within_domain_prob = 0.75`, fixed T = 180 (`HEAD`, seed 42)", vl["variable_length_false"]["full_seq_eval"]))
    out.append(row("1. `within_domain_prob = 1.0`, fixed T = 180 (`HEAD`, seed 42)", fc["p_within_1.0"]["multi_domain_eval"]))
    out.append(row("1. `within_domain_prob = 0.0`, T ~ U(90,180) (`HEAD`, seed 42)", dm["p_within_0.0"]["multi_domain_eval"]))
    if "univ_conformer_small_ge2e" in rerun:
        out.append(row("1. `within_domain_prob = 0.75`, T ~ U(90,180) (`HEAD`, seed 42)", rerun["univ_conformer_small_ge2e"]["multi_domain_eval"]))
    out.append(row("1. `within_domain_prob = 1.0`, T ~ U(90,180) (`HEAD`, seed 42)", dm["p_within_1.0"]["multi_domain_eval"]))
    xs = abl["extended_set_and_dr_ablation"]
    out.append(row("2. Pure extended-set softmax (paper Eq. 2), cosine", xs["ext_softmax_pure_eq3"]["multi_domain_eval"]))
    out.append(row("2. Blended extended-set softmax (FlaxSpeaker variant), cosine §", xs["ext_softmax_blended"]["multi_domain_eval"]))
    out.append(row("2. 3-term Dr-Vector variant (FlaxSpeaker extension) + pure extended-set", xs["dr_vectors_3term_pure_eq7"]["multi_domain_eval"]))
    out.append(row("2. 1-term down-scaled Dr-Vector variant + blended extended-set §", xs["dr_vectors_1term_blended"]["multi_domain_eval"]))
    out.append(row("3. Fixed T = 180 (released checkpoint, earlier revision), full-sequence eval §", vl["variable_length_true"]["full_seq_eval"]))
    out.append(row("3. Fixed T = 180 (`HEAD`, seed 42), full-sequence eval", vl["variable_length_false"]["full_seq_eval"]))
    if "univ_conformer_small_ge2e" in rerun:
        out.append(row("3. T ~ U(90,180) (`HEAD`, seed 42), full-sequence eval", rerun["univ_conformer_small_ge2e"]["multi_domain_eval"]))
    out.append(row("3. Fixed T = 180 (released checkpoint, earlier revision), 1.0 s (100-frame) eval §", vl["variable_length_true"]["short_1s_eval"]))
    out.append(row("3. Fixed T = 180 (`HEAD`, seed 42), 1.0 s (100-frame) eval", vl["variable_length_false"]["short_1s_eval"]))
    h = abl["training_horizon_ablation"]
    out.append(row("4. 400 steps (released checkpoint, earlier revision) §", h["steps_400"]["multi_domain_eval"]))
    if "univ_conformer_small_ge2e" in rerun:
        out.append(
            row(
                "4. 400 steps (same code revision `HEAD`, `seed=42`, `initial_loss=2.0525`)",
                rerun["univ_conformer_small_ge2e"]["multi_domain_eval"],
            )
        )
    out.append(row("4. 1,200-step run, snapshot at step 800 (LR mid-schedule; **not** an 800-step budget run)", h["steps_800"]["multi_domain_eval"]))
    h800 = r2["training_horizon_800_own_schedule"]["steps_800_own_schedule"]
    out.append(row("4. 800 steps with its own 800-step schedule (`HEAD`, seed 42)", h800["multi_domain_eval"]))
    out.append(row("4. 1,200 steps", h["steps_1200"]["multi_domain_eval"]))
    m = abl["mamba_directionality_ablation"]
    out.append(row("5. Mamba unidirectional §", m["unidirectional"]["multi_domain_eval"]))
    out.append(row("5. Mamba bidirectional", m["bidirectional"]["multi_domain_eval"]))
    out.append("")
    uni = m["unidirectional"]["multi_domain_eval"]
    bi = m["bidirectional"]["multi_domain_eval"]
    out.append(
        "Mamba 1-utt minDCF@0.01 (bidirectional vs unidirectional): "
        + ", ".join(
            f"{dom} {bi['domains_1utt'][dom]['min_dcf_001']:.4f} vs {uni['domains_1utt'][dom]['min_dcf_001']:.4f}"
            for dom in ("librispeech", "cnceleb2", "voxceleb1", "universal_pooled")
        )
        + f", macro {bi['universal_macro_avg_1utt']['min_dcf_001']:.4f} vs {uni['universal_macro_avg_1utt']['min_dcf_001']:.4f}.\n"
    )

    # Seed floor
    out.append("### Seed-noise floor (`r2_seed_and_control_results.json`)\n")
    out.append("| Model | Seeds | Pooled 1-utt EER per seed (%) | Pooled 1-utt mean ± sd | Macro 1-utt mean ± sd | Pooled 3-utt mean ± sd | Released seed-42 pooled / macro 1-utt (earlier code) | Seed 42 re-trained, current code: pooled / macro 1-utt |")
    out.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for mid, e in r2["seed_noise_floor"].items():
        ms = e["mean_std_over_seeds"]
        per = e["per_seed"]
        seeds = sorted(per, key=int)
        rel = ex[mid]["multi_domain_eval"]
        rr = rerun.get(mid)
        rr_cell = f"{f2(rr['summary']['pooled_1utt_eer'])} / {f2(rr['summary']['macro_1utt_eer'])}" if rr else "not run"
        out.append(
            f"| `{mid}` | {', '.join(seeds)} | {' / '.join(f2(per[s]['summary']['pooled_1utt_eer']) for s in seeds)} | "
            f"{f2(ms['pooled_1utt_eer']['mean'])} ± {f2(ms['pooled_1utt_eer']['std'])} | "
            f"{f2(ms['macro_1utt_eer']['mean'])} ± {f2(ms['macro_1utt_eer']['std'])} | "
            f"{f2(ms['pooled_3utt_eer']['mean'])} ± {f2(ms['pooled_3utt_eer']['std'])} | "
            f"{f2(rel['domains_1utt']['universal_pooled']['eer_percent'])} / {f2(rel['universal_macro_avg_1utt']['eer_percent'])} | "
            f"{rr_cell} |"
        )
    out.append("")
    if rerun:
        out.append("Initial / final training loss, released (earlier code) vs. seed-42 re-run (current code): "
                   + "; ".join(
                       f"`{mid}` {ex[mid]['initial_loss']:.4f} / {ex[mid]['final_loss']:.4f} vs {rr['initial_loss']:.4f} / {rr['final_loss']:.4f}"
                       for mid, rr in rerun.items())
                   + ".\n")

    # Fixed-crop seed floor + ls fixed-crop re-run
    out.append("### Fixed-crop (T = 180) same-code controls (`r3_fixed_crop_controls.json`, code `" + str(r3.get("code_commit", "")) + "`)\n")
    fcs = r3["seed_noise_floor_fixed_crop"]["univ_conformer_small_ge2e"]
    msf = fcs["mean_std_over_seeds"]
    seeds_f = sorted(fcs["per_seed"], key=int)
    out.append("| Run | Seeds | Pooled 1-utt EER per seed (%) | Pooled 1-utt mean ± sd | Macro 1-utt mean ± sd | Pooled 3-utt mean ± sd | Initial → final loss per seed |")
    out.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    out.append(
        f"| `univ_conformer_small_ge2e`, fixed T = 180 | {', '.join(seeds_f)} | "
        f"{' / '.join(f2(fcs['per_seed'][sd]['summary']['pooled_1utt_eer']) for sd in seeds_f)} | "
        f"{f2(msf['pooled_1utt_eer']['mean'])} ± {f2(msf['pooled_1utt_eer']['std'])} | "
        f"{f2(msf['macro_1utt_eer']['mean'])} ± {f2(msf['macro_1utt_eer']['std'])} | "
        f"{f2(msf['pooled_3utt_eer']['mean'])} ± {f2(msf['pooled_3utt_eer']['std'])} | "
        + " / ".join(f"{fcs['per_seed'][sd]['initial_loss']:.4f}→{fcs['per_seed'][sd]['final_loss']:.4f}" for sd in seeds_f)
        + " |"
    )
    vs = r2["seed_noise_floor"]["univ_conformer_small_ge2e"]
    msv = vs["mean_std_over_seeds"]
    seeds_v = sorted(vs["per_seed"], key=int)
    out.append(
        f"| `univ_conformer_small_ge2e`, T ~ U(90,180) | {', '.join(seeds_v)} | "
        f"{' / '.join(f2(vs['per_seed'][sd]['summary']['pooled_1utt_eer']) for sd in seeds_v)} | "
        f"{f2(msv['pooled_1utt_eer']['mean'])} ± {f2(msv['pooled_1utt_eer']['std'])} | "
        f"{f2(msv['macro_1utt_eer']['mean'])} ± {f2(msv['macro_1utt_eer']['std'])} | "
        f"{f2(msv['pooled_3utt_eer']['mean'])} ± {f2(msv['pooled_3utt_eer']['std'])} | "
        + " / ".join(f"{vs['per_seed'][sd]['initial_loss']:.4f}→{vs['per_seed'][sd]['final_loss']:.4f}" for sd in seeds_v)
        + " |"
    )
    out.append("")
    lsf = r3["ls_fixed_crop_seed42"]["ls_conformer_small_ge2e"]
    lsr = ex["ls_conformer_small_ge2e"]
    lsv = rerun.get("ls_conformer_small_ge2e")
    out.append("| `ls_conformer_small_ge2e`, seed 42 | Crop regime | Code | Initial / final loss | LS 1-utt | Pooled 1-utt [95% CI] | Pooled 3-utt |")
    out.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    p_rel = lsr["multi_domain_eval"]["domains_1utt"]["universal_pooled"]
    out.append(f"| released checkpoint | fixed T = 180 | earlier revision | {lsr['initial_loss']:.4f} / {lsr['final_loss']:.4f} | {f2(lsr['multi_domain_eval']['domains_1utt']['librispeech']['eer_percent'])} | {f2(p_rel['eer_percent'])} [{f2(p_rel['eer_ci95_low_percent'])}, {f2(p_rel['eer_ci95_high_percent'])}] | {f2(lsr['multi_domain_eval']['domains_3utt']['universal_pooled']['eer_percent'])} |")
    p_f = lsf["multi_domain_eval"]["domains_1utt"]["universal_pooled"]
    out.append(f"| re-trained | fixed T = 180 | `HEAD` | {lsf['initial_loss']:.4f} / {lsf['final_loss']:.4f} | {f2(lsf['summary']['ls_1utt_eer'])} | {f2(p_f['eer_percent'])} [{f2(p_f['eer_ci95_low_percent'])}, {f2(p_f['eer_ci95_high_percent'])}] | {f2(lsf['summary']['pooled_3utt_eer'])} |")
    if lsv:
        p_v = lsv["multi_domain_eval"]["domains_1utt"]["universal_pooled"]
        out.append(f"| re-trained | T ~ U(90,180) | `HEAD` | {lsv['initial_loss']:.4f} / {lsv['final_loss']:.4f} | {f2(lsv['summary']['ls_1utt_eer'])} | {f2(p_v['eer_percent'])} [{f2(p_v['eer_ci95_low_percent'])}, {f2(p_v['eer_ci95_high_percent'])}] | {f2(lsv['summary']['pooled_3utt_eer'])} |")
    out.append("")

    # Step-0 fingerprint
    ck = fps["checkpoints"]
    hist = fps["histogram"]
    out.append(f"### Step-0 loss fingerprint of all 44 released checkpoints (`step0_fingerprints.json`, code `{fps.get('code_commit', '')}`; tolerance {fps['tolerance']})\n")
    out.append("Histogram: " + ", ".join(f"`{k}` = {v}" for k, v in sorted(hist.items())) + ".\n")
    out.append("| Checkpoint | Loss | Released `initial_loss` | `HEAD`, fixed T = 180 | `HEAD`, T ~ U(90,180) | Match |")
    out.append("| :--- | :--- | :---: | :---: | :---: | :--- |")
    for mid, v in sorted(ck.items(), key=lambda kv: (kv[1]["match"], kv[0])):
        fx = v["head_fixed_legacy_lambda"] if v.get("legacy_lambda_used") else v["head_fixed"]
        vr = v["head_variable_legacy_lambda"] if v.get("legacy_lambda_used") else v["head_variable"]
        note = " (constant SphereFace λ = 5, pre-`9accf95` behaviour)" if v.get("legacy_lambda_used") else ""
        out.append(f"| `{mid}` | `{v['loss_type']}` | {v['released_initial_loss']:.4f} | {fx:.4f} | {vr:.4f} | `{v['match']}`{note} |")
    out.append("")

    # Matched data
    md = r2["matched_data_control"]["univ_subsampled_5788"]
    sm = md["summary"]
    e_l = ex["ls_conformer_small_ge2e"]["multi_domain_eval"]
    e_u = ex["univ_conformer_small_ge2e"]["multi_domain_eval"]
    out.append("### Matched-data control (`r2_seed_and_control_results.json`)\n")
    out.append("| Training set | Speakers / utterances | LS | CN | VOX | Macro 1-utt | Pooled 1-utt | Macro 3-utt | Pooled 3-utt |")
    out.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    def summary_of(mde):
        d1, d3 = mde["domains_1utt"], mde["domains_3utt"]
        return {
            "ls_1utt_eer": d1["librispeech"]["eer_percent"],
            "cn_1utt_eer": d1["cnceleb2"]["eer_percent"],
            "vox_1utt_eer": d1["voxceleb1"]["eer_percent"],
            "macro_1utt_eer": mde["universal_macro_avg_1utt"]["eer_percent"],
            "pooled_1utt_eer": d1["universal_pooled"]["eer_percent"],
            "macro_3utt_eer": mde["universal_macro_avg_3utt"]["eer_percent"],
            "pooled_3utt_eer": d3["universal_pooled"]["eer_percent"],
        }

    def mrow(label, spk, utt, summ):
        keys = ("ls_1utt_eer", "cn_1utt_eer", "vox_1utt_eer", "macro_1utt_eer",
                "pooled_1utt_eer", "macro_3utt_eer", "pooled_3utt_eer")
        return f"| {label} | {spk} / {utt:,} | " + " | ".join(f2(summ[k]) for k in keys) + " |"

    out.append(mrow("`ls_conformer_small_ge2e` (released; earlier code)", 232, 5788, summary_of(e_l)))
    if "ls_conformer_small_ge2e" in rerun:
        out.append(mrow("`ls_conformer_small_ge2e` (seed 42 re-trained; current code)", 232, 5788, rerun["ls_conformer_small_ge2e"]["summary"]))
    out.append(mrow("universal corpus subsampled to the LibriSpeech count (seed 42; current code)", md["num_train_speakers"], md["num_train_utterances"], sm))
    if "univ_conformer_small_ge2e" in rerun:
        out.append(mrow("`univ_conformer_small_ge2e` (seed 42 re-trained; current code)", 381, 9447, rerun["univ_conformer_small_ge2e"]["summary"]))
    out.append(mrow("`univ_conformer_small_ge2e` (released; earlier code)", 381, 9447, summary_of(e_u)))
    out.append("")

    # TFLite 180-frame verification EER & latency table
    out.append("### TFLite 180-frame (1.8 s) verification EER & latency (`benchmark_results.json`)\n")
    out.append("| Model | FP32 / INT8 size (MB) | Ratio | Primary eval set | TFLite FP32 EER (%) | TFLite INT8 EER (%) | JAX full-seq (<=500f) EER (%) | `cos(FP32, INT8)` mean (min) | CPU latency FP32 / INT8 (ms, 50-run mean) |")
    out.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    tfl_ex = bench.get("tflite_exports", {})
    for mid in (
        "univ_conformer_small_ge2e",
        "univ_ecapa_tdnn_small_ge2e",
        "univ_lstm_small_ge2e",
        "ls_conformer_small_ge2e",
        "ls_ecapa_tdnn_small_ge2e",
        "ls_lstm_small_ge2e",
    ):
        te = ex[mid].get("tflite_eval", {})
        fp_b = tfl_ex[f"{mid}_fp32"]["size_bytes"]
        in_b = tfl_ex[f"{mid}_int8"]["size_bytes"]
        out.append(
            f"| `{mid}` | {fp_b / 1e6:.2f} / {in_b / 1e6:.2f} | {fp_b / in_b:.2f}× | "
            f"`{te['primary_domain']}` | {f2(te['eer_percent'])} | {f2(te['eer_percent_int8'])} | "
            f"{f2(te['jax_full_seq_eer_percent'])} | {te['cos_fp32_int8_mean']:.6f} ({te['cos_fp32_int8_min']:.6f}) | "
            f"{f2(te['latency_ms_fp32_mean_50runs'])} / {f2(te['latency_ms_int8_mean_50runs'])} |"
        )
    out.append("")
    sys.stdout.write("\n".join(out))


if __name__ == "__main__":
    main()
