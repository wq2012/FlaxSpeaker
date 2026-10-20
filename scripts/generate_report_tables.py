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

    out.append("### Controlled ablations (`ablation_results.json`; § = copied from the main benchmark, not re-trained)\n")
    out.append("| Variant | LS 1/3-utt | CN 1/3-utt | VOX 1/3-utt | Macro 1/3-utt | Pooled 1-utt [95% CI] / 3-utt |")
    out.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    dm = abl["domain_mixing_ablation"]
    out.append(row("1. `within_domain_prob = 0.0` (pure cross-domain)", dm["p_within_0.0"]["multi_domain_eval"]))
    out.append(row("1. `within_domain_prob = 0.75` (default) §", dm["p_within_0.75"]["multi_domain_eval"]))
    out.append(row("1. `within_domain_prob = 1.0` (pure within-domain)", dm["p_within_1.0"]["multi_domain_eval"]))
    xs = abl["extended_set_and_dr_ablation"]
    out.append(row("2. Pure extended-set softmax (paper Eq. 2), cosine", xs["ext_softmax_pure_eq3"]["multi_domain_eval"]))
    out.append(row("2. Blended extended-set softmax (FlaxSpeaker variant), cosine §", xs["ext_softmax_blended"]["multi_domain_eval"]))
    out.append(row("2. 3-term Dr-Vector variant (FlaxSpeaker extension) + pure extended-set", xs["dr_vectors_3term_pure_eq7"]["multi_domain_eval"]))
    out.append(row("2. 1-term down-scaled Dr-Vector variant + blended extended-set §", xs["dr_vectors_1term_blended"]["multi_domain_eval"]))
    vl = abl["variable_length_ablation"]
    out.append(row("3. Variable length U(90,180), full-sequence eval §", vl["variable_length_true"]["full_seq_eval"]))
    out.append(row("3. Fixed length 180, full-sequence eval", vl["variable_length_false"]["full_seq_eval"]))
    out.append(row("3. Variable length, 1.0 s (100-frame) eval §", vl["variable_length_true"]["short_1s_eval"]))
    out.append(row("3. Fixed length, 1.0 s (100-frame) eval", vl["variable_length_false"]["short_1s_eval"]))
    h = abl["training_horizon_ablation"]
    out.append(row("4. 400 steps (separate run) §", h["steps_400"]["multi_domain_eval"]))
    out.append(row("4. 1,200-step run, snapshot at step 800 (LR mid-schedule; **not** an 800-step budget run)", h["steps_800"]["multi_domain_eval"]))
    h800 = r2["training_horizon_800_own_schedule"]["steps_800_own_schedule"]
    out.append(row("4. 800 steps with its own 800-step schedule (Round 2)", h800["multi_domain_eval"]))
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
    out.append("| Model | Seeds | Pooled 1-utt EER per seed (%) | Pooled 1-utt mean ± sd | Macro 1-utt mean ± sd | Pooled 3-utt mean ± sd | Released seed-42 pooled / macro 1-utt |")
    out.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for mid, e in r2["seed_noise_floor"].items():
        ms = e["mean_std_over_seeds"]
        per = e["per_seed"]
        seeds = sorted(per, key=int)
        rel = ex[mid]["multi_domain_eval"]
        out.append(
            f"| `{mid}` | {', '.join(seeds)} | {' / '.join(f2(per[s]['summary']['pooled_1utt_eer']) for s in seeds)} | "
            f"{f2(ms['pooled_1utt_eer']['mean'])} ± {f2(ms['pooled_1utt_eer']['std'])} | "
            f"{f2(ms['macro_1utt_eer']['mean'])} ± {f2(ms['macro_1utt_eer']['std'])} | "
            f"{f2(ms['pooled_3utt_eer']['mean'])} ± {f2(ms['pooled_3utt_eer']['std'])} | "
            f"{f2(rel['domains_1utt']['universal_pooled']['eer_percent'])} / {f2(rel['universal_macro_avg_1utt']['eer_percent'])} |"
        )
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

    out.append(mrow("`ls_conformer_small_ge2e` (LibriSpeech only, seed 42)", 232, 5788, summary_of(e_l)))
    out.append(mrow("universal corpus subsampled to the LibriSpeech count (seed 42)", md["num_train_speakers"], md["num_train_utterances"], sm))
    out.append(mrow("`univ_conformer_small_ge2e` (full universal corpus, seed 42)", 381, 9447, summary_of(e_u)))
    out.append("")
    sys.stdout.write("\n".join(out))


if __name__ == "__main__":
    main()
