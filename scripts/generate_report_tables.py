#!/usr/bin/env python3
"""Regenerate the number-bearing tables of README.md and RESEARCH_REPORT.md from JSON archives.

Prints Markdown to stdout so that no benchmark number in README.md or RESEARCH_REPORT.md
is typed by hand:

    python scripts/generate_report_tables.py --section all > /tmp/report_tables.md
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PM = os.path.join(REPO_ROOT, "pretrained_models")

FLAGSHIP_ORDER = [
    "univ_full_conformer_large_dr_vectors",
    "univ_full_conformer_large_pfas",
    "univ_full_ecapa_c1024_arcface",
    "univ_full_conformer_large_arcface",
]


def load(name: str):
    with open(os.path.join(PM, name), "r", encoding="utf-8") as f:
        return json.load(f)


def f2(x: float) -> str:
    return f"{x:.2f}"


def f4(x: float) -> str:
    return f"{x:.4f}"


def d1(e: dict, dom: str) -> dict:
    return e["multi_domain_eval"]["domains_1utt"][dom]


def d3(e: dict, dom: str) -> dict:
    return e["multi_domain_eval"]["domains_3utt"][dom]


def mac(e: dict, n: int) -> dict:
    return e["multi_domain_eval"][f"universal_macro_avg_{n}utt"]


def bmin(val: float, vals: list[float], suffix: str = "%") -> str:
    s = f"{f2(val)}{suffix}"
    return f"**{s}**" if round(val, 4) == round(min(vals), 4) else s


def render_flagship_tables(flagship: dict, pfas_sweep: dict) -> list[str]:
    out: list[str] = []
    fm = flagship["models"]
    fus = flagship["heterogeneous_fusion"]
    fus2 = fus["official_benchmarks"]
    fus3 = fus["multi_enroll_3way_fusion"]

    out.append("<!-- BEGIN README_FLAGSHIP_TABLE -->")
    out.append(
        "| Hugging Face Repository | Backbone / Head / Objective | Params | Steps | "
        "VoxCeleb1-O Official (`37,611` trials) Raw / AS-Norm / QMF EER (%) | "
        "CN-Celeb-E Official (`3.48M` trials) Single Raw / QMF / Multi-Enroll EER (%) | "
        "LibriSpeech `test-clean` (`10,000` trials) 1-Utt Raw / QMF / 3-Utt EER (%) | "
        "Controlled 3-Domain Pooled 1-Utt / 3-Utt EER (%) | Controlled Macro-Avg 1-Utt / 3-Utt EER (%) |"
    )
    out.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    min_vx_qmf = min(round(fm[mid]["official_benchmarks"]["voxceleb1_o"]["qmf"]["eer_percent"], 2) for mid in FLAGSHIP_ORDER)
    min_cn_qmf = min(round(fm[mid]["official_benchmarks"]["cnceleb_e"]["qmf"]["eer_percent"], 2) for mid in FLAGSHIP_ORDER)
    min_ls_qmf = min(round(fm[mid]["official_benchmarks"]["librispeech_test_clean"]["qmf"]["eer_percent"], 2) for mid in FLAGSHIP_ORDER)
    min_ls_3 = min(round(fm[mid]["official_benchmarks"]["librispeech_test_clean"]["multi_enroll_3utt"]["eer_percent"], 2) for mid in FLAGSHIP_ORDER)
    for mid in FLAGSHIP_ORDER:
        m = fm[mid]
        ob = m["official_benchmarks"]
        vx = ob["voxceleb1_o"]
        cn = ob["cnceleb_e"]
        ls = ob["librispeech_test_clean"]
        vx_as = f"{f2(vx['as_norm']['eer_percent'])}%" if "as_norm" in vx else "—"
        vx_qmf = f"**{f2(vx['qmf']['eer_percent'])}%**" if round(vx["qmf"]["eer_percent"], 2) == min_vx_qmf else f"{f2(vx['qmf']['eer_percent'])}%"
        cn_me = cn["multi_enroll_map"].get("as_norm", cn["multi_enroll_map"])["eer_percent"]
        cn_qmf = f"**{f2(cn['qmf']['eer_percent'])}%**" if round(cn["qmf"]["eer_percent"], 2) == min_cn_qmf else f"{f2(cn['qmf']['eer_percent'])}%"
        ls_qmf = f"**{f2(ls['qmf']['eer_percent'])}%**" if round(ls["qmf"]["eer_percent"], 2) == min_ls_qmf else f"{f2(ls['qmf']['eer_percent'])}%"
        ls_3 = f"**{f2(ls['multi_enroll_3utt']['eer_percent'])}%**" if round(ls["multi_enroll_3utt"]["eer_percent"], 2) == min_ls_3 else f"{f2(ls['multi_enroll_3utt']['eer_percent'])}%"
        p1 = d1(m, "universal_pooled")["eer_percent"]
        p3_raw = d3(m, "universal_pooled")["eer_percent"]
        p3_best = d3(m, "universal_pooled").get("as_norm", {}).get("eer_percent", p3_raw)
        p3_disp = min(p3_raw, p3_best)
        m1_raw = mac(m, 1)["eer_percent"]
        m3_raw = mac(m, 3)["eer_percent"]
        out.append(
            f"| [`wq2012/{mid}`](https://huggingface.co/wq2012/{mid}) | "
            f"`{m['backbone']}` (`{m['size_variant']}`) / `{m['pooling_type']}` / `{m['loss_type']}` (`{m['scoring_type']}`) | "
            f"`{int(m['num_parameters']):,}` | `{int(m['num_steps']):,}` | "
            f"{f2(vx['eer_percent'])}% / {vx_as} / {vx_qmf} | "
            f"{f2(cn['eer_percent'])}% / {cn_qmf} / {f2(cn_me)}% | "
            f"{f2(ls['eer_percent'])}% / {ls_qmf} / {ls_3} | "
            f"{f2(p1)}% / {f2(p3_disp)}% (raw {f2(p3_raw)}%) | "
            f"{f2(m1_raw)}% / {f2(m3_raw)}% |"
        )

    if "two_way_official_benchmarks" in fus:
        fus_2w = fus["two_way_official_benchmarks"]
        w2 = fus.get("two_way_weights_selected_on_val", [0.5, 0.5])
        out.append(
            f"| **2-Way Score Fusion (`{w2[0]:.2g}·Conformer-L + {w2[1]:.2g}·ECAPA-c1024`)** | "
            f"Heterogeneous Cosine + AS-Norm + QMF | `21,906,500` | — | "
            f"— / {f2(fus_2w['voxceleb1_o']['eer_percent'])}% / {f2(fus_2w['voxceleb1_o']['qmf']['eer_percent'])}% | "
            f"— / {f2(fus_2w['cnceleb_e']['qmf']['eer_percent'])}% / — | "
            f"{f2(fus_2w['librispeech_test_clean']['eer_percent'])}% / {f2(fus_2w['librispeech_test_clean']['qmf']['eer_percent'])}% / — | "
            f"— | — |"
        )
        out.append(
            f"| **4-Way Flagship Fusion (`Conf-L + ECAPA + PFAS + DrV`)** | "
            f"Heterogeneous 4-Way AS-Norm + QMF | `36,367,570` | — | "
            f"— / **{f2(fus2['voxceleb1_o']['eer_percent'])}%** / **{f2(fus2['voxceleb1_o']['qmf']['eer_percent'])}%** | "
            f"— / **{f2(fus2['cnceleb_e']['qmf']['eer_percent'])}%** / — | "
            f"**{f2(fus2['librispeech_test_clean']['eer_percent'])}%** / **{f2(fus2['librispeech_test_clean']['qmf']['eer_percent'])}%** / — | "
            f"— | — |"
        )
    else:
        out.append(
            f"| **2-Way Score Fusion (`0.1·Conformer-L + 0.9·ECAPA-c1024`)** | "
            f"Heterogeneous Cosine + AS-Norm + QMF | `21,906,500` | — | "
            f"— / {f2(fus2['voxceleb1_o']['eer_percent'])}% / **{f2(fus2['voxceleb1_o']['qmf']['eer_percent'])}%** | "
            f"— / {f2(fus2['cnceleb_e']['qmf']['eer_percent'])}% / — | "
            f"{f2(fus2['librispeech_test_clean']['eer_percent'])}% / {f2(fus2['librispeech_test_clean']['qmf']['eer_percent'])}% / — | "
            f"— | — |"
        )
    out.append(
        f"| **3-Way Multi-Enroll Fusion (`0.4·Conf-L + 0.4·ECAPA + 0.2·PFAS`)** | "
        f"Cosine AS-Norm + Frame-Level PFAS | `29,218,827` | — | "
        f"— | — / — / **{f2(fus3['cnceleb_e_multi_enroll_map']['eer_percent'])}%** (`AUC {f4(fus3['cnceleb_e_multi_enroll_map']['roc_auc'])}`) | "
        f"— / — / **{f2(fus3['librispeech_test_clean_3utt']['eer_percent'])}%** (`AUC {f4(fus3['librispeech_test_clean_3utt']['roc_auc'])}`) | "
        f"— | — |"
    )
    out.append("<!-- END README_FLAGSHIP_TABLE -->\n")

    out.append("<!-- BEGIN PFAS_SWEEP_TABLE -->")
    out.append(
        "| Rank | PFAS Candidate | Attention (`K × (d_k, d_v)`) | Output Dim | Anchor (`λ_loc`) | Loss (`Pure`) | "
        "Params | Val 1-Utt EER (%) [95% CI] | Val 3-Utt EER (%) [95% CI] | Composite Val EER (%) |"
    )
    out.append("| :---: | :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: |")
    cands = pfas_sweep["candidates"]
    for item in pfas_sweep["ranking"]:
        cname = item["name"]
        c = cands[cname]
        sp = c["spec"]
        v1 = c["val_1utt_metrics"]
        v3 = c["val_3utt_metrics"]
        anchor_str = f"`True` (`{sp['local_value_scale']:.2f}`)" if sp["use_global_anchor"] else "`False`"
        loss_str = f"`{sp['loss_type']}` (`{sp['pure_extended_set']}`)"
        bold = "**" if item["rank"] == 1 else ""
        out.append(
            f"| {bold}{item['rank']}{bold} | `{cname}` | "
            f"`{sp['attention_type']}` (`{sp['num_keys']}×({sp['key_dim']},{sp['value_dim']})`) | "
            f"`{c['representation_dim']}` | {anchor_str} | {loss_str} | `{int(c['num_parameters']):,}` | "
            f"{bold}{f2(v1['eer_percent'])}%{bold} [{f2(v1['eer_ci95_low_percent'])}, {f2(v1['eer_ci95_high_percent'])}] | "
            f"{bold}{f2(v3['eer_percent'])}%{bold} [{f2(v3['eer_ci95_low_percent'])}, {f2(v3['eer_ci95_high_percent'])}] | "
            f"{bold}{f2(c['composite_val_eer_percent'])}%{bold} |"
        )
    out.append("<!-- END PFAS_SWEEP_TABLE -->\n")
    return out


def render_readme_tables(ex: dict) -> list[str]:
    out: list[str] = []
    univ_ids = sorted(
        [k for k in ex if k.startswith("univ_") and not k.startswith("univ_full_")],
        key=lambda k: (d1(ex[k], "universal_pooled")["eer_percent"], mac(ex[k], 1)["eer_percent"]),
    )

    out.append("<!-- BEGIN README_UNIV_TABLE -->")
    out.append(
        "| Hugging Face Repository | Steps | Backbone / Size | Pooling / Loss / Scoring | Params | "
        "LS 1-Utt / 3-Utt EER (%) | CN 1-Utt / 3-Utt EER (%) | VOX 1-Utt / 3-Utt EER (%) | "
        "Macro-Avg 1-Utt / 3-Utt EER (%) | Pooled 1-Utt / 3-Utt EER (%) | Macro 1-Utt / 3-Utt minDCF@0.01 |"
    )
    out.append("| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    ls1_v = [d1(ex[k], "librispeech")["eer_percent"] for k in univ_ids]
    ls3_v = [d3(ex[k], "librispeech")["eer_percent"] for k in univ_ids]
    cn1_v = [d1(ex[k], "cnceleb2")["eer_percent"] for k in univ_ids]
    cn3_v = [d3(ex[k], "cnceleb2")["eer_percent"] for k in univ_ids]
    vx1_v = [d1(ex[k], "voxceleb1")["eer_percent"] for k in univ_ids]
    vx3_v = [d3(ex[k], "voxceleb1")["eer_percent"] for k in univ_ids]
    mc1_v = [mac(ex[k], 1)["eer_percent"] for k in univ_ids]
    mc3_v = [mac(ex[k], 3)["eer_percent"] for k in univ_ids]
    pl1_v = [d1(ex[k], "universal_pooled")["eer_percent"] for k in univ_ids]
    pl3_v = [d3(ex[k], "universal_pooled")["eer_percent"] for k in univ_ids]
    dcf1_v = [mac(ex[k], 1)["min_dcf_001"] for k in univ_ids]
    dcf3_v = [mac(ex[k], 3)["min_dcf_001"] for k in univ_ids]

    for mid in univ_ids:
        e = ex[mid]
        dcf1 = mac(e, 1)["min_dcf_001"]
        dcf3 = mac(e, 3)["min_dcf_001"]
        dcf1_s = f"**`{f4(dcf1)}`**" if round(dcf1, 4) == round(min(dcf1_v), 4) else f"`{f4(dcf1)}`"
        dcf3_s = f"**`{f4(dcf3)}`**" if round(dcf3, 4) == round(min(dcf3_v), 4) else f"`{f4(dcf3)}`"
        out.append(
            f"| [`wq2012/{mid}`](https://huggingface.co/wq2012/{mid}) | `{e['num_steps']:,}` | "
            f"`{e['backbone']}` (`{e['size_variant']}`) | "
            f"`{e['pooling_type']}` / `{e['loss_type']}` / `{e['scoring_type']}` | "
            f"`{int(e['num_parameters']):,}` | "
            f"{bmin(d1(e, 'librispeech')['eer_percent'], ls1_v)} / {bmin(d3(e, 'librispeech')['eer_percent'], ls3_v)} | "
            f"{bmin(d1(e, 'cnceleb2')['eer_percent'], cn1_v)} / {bmin(d3(e, 'cnceleb2')['eer_percent'], cn3_v)} | "
            f"{bmin(d1(e, 'voxceleb1')['eer_percent'], vx1_v)} / {bmin(d3(e, 'voxceleb1')['eer_percent'], vx3_v)} | "
            f"{bmin(mac(e, 1)['eer_percent'], mc1_v)} / {bmin(mac(e, 3)['eer_percent'], mc3_v)} | "
            f"{bmin(d1(e, 'universal_pooled')['eer_percent'], pl1_v)} / {bmin(d3(e, 'universal_pooled')['eer_percent'], pl3_v)} | "
            f"{dcf1_s} / {dcf3_s} |"
        )
    out.append("<!-- END README_UNIV_TABLE -->\n")

    single_ids = sorted(
        [k for k in ex if not k.startswith("univ_")],
        key=lambda k: (
            {"librispeech": 0, "cnceleb2": 1, "voxceleb1": 2}[ex[k]["train_dataset"]],
            d1(ex[k], ex[k]["train_dataset"])["eer_percent"],
        ),
    )
    out.append("<!-- BEGIN README_SINGLE_TABLE -->")
    out.append(
        "| Hugging Face Repository | Train Domain (`Steps`) | Backbone / Size | Pooling / Loss / Scoring | Params | "
        "LS 1-Utt / 3-Utt EER (%) | CN 1-Utt / 3-Utt EER (%) | VOX 1-Utt / 3-Utt EER (%) | "
        "Macro-Avg 1-Utt / 3-Utt EER (%) | Pooled 1-Utt / 3-Utt EER (%) |"
    )
    out.append("| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    s_ls1 = [d1(ex[k], "librispeech")["eer_percent"] for k in single_ids]
    s_ls3 = [d3(ex[k], "librispeech")["eer_percent"] for k in single_ids]
    s_cn1 = [d1(ex[k], "cnceleb2")["eer_percent"] for k in single_ids]
    s_cn3 = [d3(ex[k], "cnceleb2")["eer_percent"] for k in single_ids]
    s_vx1 = [d1(ex[k], "voxceleb1")["eer_percent"] for k in single_ids]
    s_vx3 = [d3(ex[k], "voxceleb1")["eer_percent"] for k in single_ids]
    s_mc1 = [mac(ex[k], 1)["eer_percent"] for k in single_ids]
    s_mc3 = [mac(ex[k], 3)["eer_percent"] for k in single_ids]
    s_pl1 = [d1(ex[k], "universal_pooled")["eer_percent"] for k in single_ids]
    s_pl3 = [d3(ex[k], "universal_pooled")["eer_percent"] for k in single_ids]
    for mid in single_ids:
        e = ex[mid]
        out.append(
            f"| [`wq2012/{mid}`](https://huggingface.co/wq2012/{mid}) | `{e['train_dataset']}` (`{e['num_steps']:,}`) | "
            f"`{e['backbone']}` (`{e['size_variant']}`) | "
            f"`{e['pooling_type']}` / `{e['loss_type']}` / `{e['scoring_type']}` | "
            f"`{int(e['num_parameters']):,}` | "
            f"{bmin(d1(e, 'librispeech')['eer_percent'], s_ls1)} / {bmin(d3(e, 'librispeech')['eer_percent'], s_ls3)} | "
            f"{bmin(d1(e, 'cnceleb2')['eer_percent'], s_cn1)} / {bmin(d3(e, 'cnceleb2')['eer_percent'], s_cn3)} | "
            f"{bmin(d1(e, 'voxceleb1')['eer_percent'], s_vx1)} / {bmin(d3(e, 'voxceleb1')['eer_percent'], s_vx3)} | "
            f"{bmin(mac(e, 1)['eer_percent'], s_mc1)} / {bmin(mac(e, 3)['eer_percent'], s_mc3)} | "
            f"{bmin(d1(e, 'universal_pooled')['eer_percent'], s_pl1)} / {bmin(d3(e, 'universal_pooled')['eer_percent'], s_pl3)} |"
        )
    out.append("<!-- END README_SINGLE_TABLE -->\n")
    return out


def render_report_studies(ex: dict) -> list[str]:
    out: list[str] = []
    bb_ids = [
        "univ_lstm_small_ge2e",
        "univ_resnet_small_ge2e",
        "univ_conformer_small_ge2e",
        "univ_ecapa_tdnn_small_ge2e",
        "univ_transformer_small_ge2e",
        "univ_mamba_small_ge2e",
    ]
    out.append("<!-- BEGIN STUDY_5_TABLES -->")
    out.append("### 5.1 Universal Backbones — 1-Utterance Verification Results\n")
    out.append(
        "| Hugging Face Repository | Backbone (`Pooling`) | Params | LibriSpeech EER (%) [95% CI] | "
        "CN-Celeb2 EER (%) [95% CI] | VoxCeleb1 EER (%) [95% CI] | Universal Macro-Avg EER (%) | "
        "Universal Pooled EER (%) [95% CI] | Pooled ROC-AUC |"
    )
    out.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for mid in sorted(bb_ids, key=lambda k: d1(ex[k], "universal_pooled")["eer_percent"]):
        e = ex[mid]
        ls, cn, vx, pl = d1(e, "librispeech"), d1(e, "cnceleb2"), d1(e, "voxceleb1"), d1(e, "universal_pooled")
        out.append(
            f"| **[`wq2012/{mid}`](https://huggingface.co/wq2012/{mid})** | `{e['backbone']}` (`{e['pooling_type']}`) | "
            f"{int(e['num_parameters']):,} | "
            f"{f2(ls['eer_percent'])}% [{f2(ls['eer_ci95_low_percent'])}, {f2(ls['eer_ci95_high_percent'])}] | "
            f"{f2(cn['eer_percent'])}% [{f2(cn['eer_ci95_low_percent'])}, {f2(cn['eer_ci95_high_percent'])}] | "
            f"{f2(vx['eer_percent'])}% [{f2(vx['eer_ci95_low_percent'])}, {f2(vx['eer_ci95_high_percent'])}] | "
            f"{f2(mac(e, 1)['eer_percent'])}% | "
            f"{f2(pl['eer_percent'])}% [{f2(pl['eer_ci95_low_percent'])}, {f2(pl['eer_ci95_high_percent'])}] | "
            f"{f4(pl['roc_auc'])} |"
        )
    out.append("\n### 5.2 Universal Backbones — 3-Utterance Multi-Enrollment Verification Results\n")
    out.append(
        "| Hugging Face Repository | Backbone (`Pooling`) | LibriSpeech 3-Utt EER (%) | CN-Celeb2 3-Utt EER (%) | "
        "VoxCeleb1 3-Utt EER (%) | Universal Macro-Avg 3-Utt EER (%) | Universal Pooled 3-Utt EER (%) [95% CI] | "
        "Macro 3-Utt minDCF@0.01 | Macro 3-Utt ROC-AUC |"
    )
    out.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for mid in sorted(bb_ids, key=lambda k: d3(ex[k], "universal_pooled")["eer_percent"]):
        e = ex[mid]
        ls3, cn3, vx3, pl3 = d3(e, "librispeech"), d3(e, "cnceleb2"), d3(e, "voxceleb1"), d3(e, "universal_pooled")
        m3 = mac(e, 3)
        out.append(
            f"| **[`wq2012/{mid}`](https://huggingface.co/wq2012/{mid})** | `{e['backbone']}` (`{e['pooling_type']}`) | "
            f"{f2(ls3['eer_percent'])}% | {f2(cn3['eer_percent'])}% | {f2(vx3['eer_percent'])}% | "
            f"{f2(m3['eer_percent'])}% | "
            f"{f2(pl3['eer_percent'])}% [{f2(pl3['eer_ci95_low_percent'])}, {f2(pl3['eer_ci95_high_percent'])}] | "
            f"{f4(m3['min_dcf_001'])} | {f4(m3['roc_auc'])} |"
        )
    out.append("<!-- END STUDY_5_TABLES -->\n")

    obj_ids = [
        k for k in ex
        if k.startswith("univ_conformer_small_")
    ]
    out.append("<!-- BEGIN STUDY_6_TABLES -->")
    out.append("### 6.1 Universal Losses & Scoring — 1-Utterance Verification\n")
    out.append(
        "| Hugging Face Repository | Loss Function | Scoring | Pooling | LibriSpeech EER (%) | CN-Celeb2 EER (%) | "
        "VoxCeleb1 EER (%) | Macro-Avg 1-Utt EER (%) | Pooled 1-Utt EER (%) [95% CI] | "
        "Macro 1-Utt minDCF@0.01 | Pooled 1-Utt ROC-AUC |"
    )
    out.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for mid in sorted(obj_ids, key=lambda k: (d1(ex[k], "universal_pooled")["eer_percent"], mac(ex[k], 1)["eer_percent"])):
        e = ex[mid]
        pl = d1(e, "universal_pooled")
        out.append(
            f"| **[`wq2012/{mid}`](https://huggingface.co/wq2012/{mid})** | `{e['loss_type']}` | `{e['scoring_type']}` | "
            f"`{e['pooling_type']}` | {f2(d1(e, 'librispeech')['eer_percent'])}% | {f2(d1(e, 'cnceleb2')['eer_percent'])}% | "
            f"{f2(d1(e, 'voxceleb1')['eer_percent'])}% | {f2(mac(e, 1)['eer_percent'])}% | "
            f"{f2(pl['eer_percent'])}% [{f2(pl['eer_ci95_low_percent'])}, {f2(pl['eer_ci95_high_percent'])}] | "
            f"{f4(mac(e, 1)['min_dcf_001'])} | {f4(pl['roc_auc'])} |"
        )
    out.append("\n### 6.2 Universal Losses & Scoring — 3-Utterance Multi-Enrollment Verification\n")
    out.append(
        "| Hugging Face Repository | Loss Function | Scoring | LibriSpeech 3-Utt EER (%) | CN-Celeb2 3-Utt EER (%) | "
        "VoxCeleb1 3-Utt EER (%) | Macro-Avg 3-Utt EER (%) | Pooled 3-Utt EER (%) [95% CI] | "
        "Macro 3-Utt minDCF@0.01 | Pooled 3-Utt ROC-AUC |"
    )
    out.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for mid in sorted(obj_ids, key=lambda k: (d3(ex[k], "universal_pooled")["eer_percent"], mac(ex[k], 3)["eer_percent"])):
        e = ex[mid]
        pl3 = d3(e, "universal_pooled")
        out.append(
            f"| **[`wq2012/{mid}`](https://huggingface.co/wq2012/{mid})** | `{e['loss_type']}` | `{e['scoring_type']}` | "
            f"{f2(d3(e, 'librispeech')['eer_percent'])}% | {f2(d3(e, 'cnceleb2')['eer_percent'])}% | "
            f"{f2(d3(e, 'voxceleb1')['eer_percent'])}% | {f2(mac(e, 3)['eer_percent'])}% | "
            f"{f2(pl3['eer_percent'])}% [{f2(pl3['eer_ci95_low_percent'])}, {f2(pl3['eer_ci95_high_percent'])}] | "
            f"{f4(mac(e, 3)['min_dcf_001'])} | {f4(pl3['roc_auc'])} |"
        )
    out.append("<!-- END STUDY_6_TABLES -->\n")

    scale_ids = [
        "univ_conformer_tiny_ge2e", "univ_conformer_small_ge2e", "univ_conformer_base_ge2e",
        "univ_ecapa_tdnn_tiny_ge2e", "univ_ecapa_tdnn_small_ge2e", "univ_ecapa_tdnn_base_ge2e",
        "ls_conformer_tiny_ge2e", "ls_conformer_small_ge2e", "ls_conformer_base_ge2e",
        "ls_ecapa_tdnn_tiny_ge2e", "ls_ecapa_tdnn_small_ge2e", "ls_ecapa_tdnn_base_ge2e",
    ]
    out.append("<!-- BEGIN STUDY_7_TABLE -->")
    out.append(
        "| Hugging Face Repository | Train (`Steps`) | Backbone & Size | Params | LibriSpeech 1-Utt EER (%) | "
        "CN-Celeb2 1-Utt EER (%) | VoxCeleb1 1-Utt EER (%) | Macro-Avg 1-Utt EER (%) | "
        "Pooled 1-Utt EER (%) [95% CI] | Macro-Avg 3-Utt EER (%) | Pooled 3-Utt EER (%) [95% CI] |"
    )
    out.append("| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for mid in scale_ids:
        e = ex[mid]
        pl1, pl3 = d1(e, "universal_pooled"), d3(e, "universal_pooled")
        out.append(
            f"| **[`wq2012/{mid}`](https://huggingface.co/wq2012/{mid})** | `{e['train_dataset']}` (`{e['num_steps']:,}`) | "
            f"`{e['backbone']}-{e['size_variant']}` | {int(e['num_parameters']):,} | "
            f"{f2(d1(e, 'librispeech')['eer_percent'])}% | {f2(d1(e, 'cnceleb2')['eer_percent'])}% | "
            f"{f2(d1(e, 'voxceleb1')['eer_percent'])}% | {f2(mac(e, 1)['eer_percent'])}% | "
            f"{f2(pl1['eer_percent'])}% [{f2(pl1['eer_ci95_low_percent'])}, {f2(pl1['eer_ci95_high_percent'])}] | "
            f"{f2(mac(e, 3)['eer_percent'])}% | "
            f"{f2(pl3['eer_percent'])}% [{f2(pl3['eer_ci95_low_percent'])}, {f2(pl3['eer_ci95_high_percent'])}] |"
        )
    out.append("<!-- END STUDY_7_TABLE -->\n")
    return out


def render_section10_and_11(bench: dict, abl: dict, vox: dict, r2: dict, fps: dict) -> list[str]:
    ex = bench["experiments"]
    out: list[str] = []

    # Vox1-O subset table
    out.append(f"### VoxCeleb1-O 10-speaker subset (all {len(vox['models'])} models)\n")
    out.append(
        f"Protocol: `{vox['protocol']}` — {vox['num_trials']:,} trials "
        f"({vox['num_target_trials']:,} target / {vox['num_impostor_trials']:,} impostor, "
        f"{100 * vox['num_target_trials'] / vox['num_trials']:.1f}% target), {vox['num_unique_utterances']} utterances "
        f"from {len(vox['eval_speakers'])} held-out speakers (`{'`, `'.join(vox['eval_speakers'])}`), "
        f"VAD {vox['vad_energy_db_threshold']:.0f} dB (matched to training). "
        f"**Not comparable to published VoxCeleb1-O numbers (see Section 8 for Official 37,611-trial VoxCeleb1-O).**\n"
    )
    out.append("| Model | Subset EER (%) [95% CI] | minDCF@0.01 | ROC-AUC | Pooled 1-utt EER (%) |")
    out.append("| :--- | :---: | :---: | :---: | :---: |")
    for mid, m in sorted(vox["models"].items(), key=lambda kv: kv[1]["eer_percent"]):
        pooled = d1(ex[mid], "universal_pooled")["eer_percent"]
        out.append(
            f"| `{mid}` | {f2(m['eer_percent'])} [{f2(m['eer_ci95_low_percent'])}, {f2(m['eer_ci95_high_percent'])}] | "
            f"{m['min_dcf_001']:.4f} | {m['roc_auc']:.4f} | {f2(pooled)} |"
        )
    out.append("")

    # Ablations
    def row(label: str, mde: dict) -> str:
        d_1, d_3 = mde["domains_1utt"], mde["domains_3utt"]
        p1 = d_1["universal_pooled"]
        return (
            f"| {label} | {f2(d_1['librispeech']['eer_percent'])} / {f2(d_3['librispeech']['eer_percent'])} | "
            f"{f2(d_1['cnceleb2']['eer_percent'])} / {f2(d_3['cnceleb2']['eer_percent'])} | "
            f"{f2(d_1['voxceleb1']['eer_percent'])} / {f2(d_3['voxceleb1']['eer_percent'])} | "
            f"{f2(mde['universal_macro_avg_1utt']['eer_percent'])} / {f2(mde['universal_macro_avg_3utt']['eer_percent'])} | "
            f"{f2(p1['eer_percent'])} [{f2(p1['eer_ci95_low_percent'])}, {f2(p1['eer_ci95_high_percent'])}] / "
            f"{f2(d_3['universal_pooled']['eer_percent'])} |"
        )

    out.append(
        "### Controlled ablations (`ablation_results.json`, `r2_seed_and_control_results.json`; "
        "§ = matched converged v0.5.0 checkpoint in `benchmark_results.json`, 4,600 steps on Universal / "
        "2,400 steps on LibriSpeech, seed 42, variable-length crops `T ~ U(90, 180)`)\n"
    )
    out.append("| Variant | LS 1/3-utt | CN 1/3-utt | VOX 1/3-utt | Macro 1/3-utt | Pooled 1-utt [95% CI] / 3-utt |")
    out.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

    dm = abl["domain_mixing_ablation"]
    out.append(row("1. Stratified batching: `within_domain_prob = 0.00` (pure cross-domain, 4,600 steps)", dm["p_within_0.0"]["multi_domain_eval"]))
    out.append(row("1. Stratified batching: `within_domain_prob = 0.75` (default hybrid, 4,600 steps) §", dm["p_within_0.75"]["multi_domain_eval"]))
    out.append(row("1. Stratified batching: `within_domain_prob = 1.00` (pure within-domain, 4,600 steps)", dm["p_within_1.0"]["multi_domain_eval"]))

    xs = abl["extended_set_and_dr_ablation"]
    st = abl.get("studies", {})
    out.append(row("2. Pure extended-set softmax (paper Eq. 2), cosine (4,600 steps)", xs["ext_softmax_pure_eq3"]["multi_domain_eval"]))
    out.append(row("2. Blended extended-set softmax (FlaxSpeaker BXS variant), cosine (4,600 steps) §", xs["ext_softmax_blended"]["multi_domain_eval"]))
    out.append(row("2. 3-term Dr-Vector variant (`[c, c^2, sqrt(max(c,0))]`) + pure extended-set (4,600 steps)", xs["dr_vectors_3term_pure_eq7"]["multi_domain_eval"]))
    out.append(row("2. 1-term down-scaled Dr-Vector variant + blended extended-set (4,600 steps) §", xs["dr_vectors_1term_blended"]["multi_domain_eval"]))
    if "dr_vector_terms_and_contrast_ablation" in st:
        dvc = st["dr_vector_terms_and_contrast_ablation"]
        out.append(row("2. 3-term Dr-Vector symmetric skip + blended extended-set (4,600 steps)", dvc["dr_vector_3term_sym_skip"]["multi_domain_eval"]))
        out.append(row("2. Hybrid `ge2e_contrast` surrogate (4,600 steps) §", dvc["ge2e_contrast_stabilized"]["multi_domain_eval"]))
    if "pfas_sn_attention_and_anchor_ablation" in st:
        pfa = st["pfas_sn_attention_and_anchor_ablation"]
        out.append(row("2b. PFAS-SN `shared_nonlinear` + global ATP anchor (4,600 steps) §", pfa["pfas_sn_mlp_with_anchor"]["multi_domain_eval"]))
        out.append(row("2b. PFAS-SN `shared_nonlinear` without global ATP anchor (4,600 steps)", pfa["pfas_sn_mlp_no_anchor"]["multi_domain_eval"]))
        out.append(row("2b. PFAS-SN `dot` attention + global ATP anchor (4,600 steps)", pfa["pfas_sn_dot_with_anchor"]["multi_domain_eval"]))
        out.append(row("2b. PFAS-SN `maxpool` + global ATP anchor (4,600 steps)", pfa["pfas_sn_maxpool_with_anchor"]["multi_domain_eval"]))

    vl = abl["variable_length_ablation"]
    out.append(row("3. Variable crops `T ~ U(90, 180)` (4,600 steps), full-sequence (`<=500f`) eval §", vl["variable_length_true"]["full_seq_eval"]))
    out.append(row("3. Fixed crops `T = 180` (4,600 steps), full-sequence (`<=500f`) eval", vl["variable_length_false"]["full_seq_eval"]))
    out.append(row("3. Variable crops `T ~ U(90, 180)` (4,600 steps), short 1.0 s (`100f`) eval §", vl["variable_length_true"]["short_1s_eval"]))
    out.append(row("3. Fixed crops `T = 180` (4,600 steps), short 1.0 s (`100f`) eval", vl["variable_length_false"]["short_1s_eval"]))

    own_h = r2.get("own_schedule_horizon", {})
    for steps_k, label_s in (
        ("steps_400", "400 steps (own 400-step cosine schedule)"),
        ("steps_800", "800 steps (own 800-step cosine schedule)"),
        ("steps_1200", "1,200 steps (own 1,200-step cosine schedule)"),
        ("steps_2400", "2,400 steps (own 2,400-step cosine schedule)"),
        ("steps_4600", "4,600 steps (`S_safe`, own 4,600-step cosine schedule) §"),
    ):
        if steps_k in own_h:
            out.append(row(f"4. Horizon: {label_s}", own_h[steps_k]["multi_domain_eval"]))

    m_ls = abl["mamba_directionality_ablation"]
    out.append(row("5a. Mamba unidirectional on LibriSpeech (2,400 steps) §", m_ls["unidirectional"]["multi_domain_eval"]))
    out.append(row("5a. Mamba bidirectional on LibriSpeech (2,400 steps)", m_ls["bidirectional"]["multi_domain_eval"]))
    if "mamba_direction_ablation" in st:
        m_u = st["mamba_direction_ablation"]
        out.append(row("5b. Mamba unidirectional on Universal (4,600 steps) §", m_u["mamba_unidirectional"]["multi_domain_eval"]))
        out.append(row("5b. Mamba bidirectional on Universal (4,600 steps)", m_u["mamba_bidirectional"]["multi_domain_eval"]))

    if "ecapa_norm_order_ablation" in st:
        ec = st["ecapa_norm_order_ablation"]
        out.append(row("6a. ECAPA-TDNN `BatchNorm` after activation (v0.5.0 default, 4,600 steps) §", ec["ecapa_bn_style_after_act"]["multi_domain_eval"]))
        out.append(row("6a. ECAPA-TDNN `LayerNorm` before activation (legacy v0.4.0 order, 4,600 steps)", ec["ecapa_ln_before_act"]["multi_domain_eval"]))
    if "vad_and_agc_ablation" in st:
        va = st["vad_and_agc_ablation"]
        out.append(row("6b. Frontend VAD 26 dB + peak AGC (default, 4,600 steps) §", va["vad_26db_with_agc"]["multi_domain_eval"]))
        out.append(row("6b. Frontend VAD 26 dB without AGC (4,600 steps)", va["vad_26db_no_agc"]["multi_domain_eval"]))
        out.append(row("6b. Frontend VAD 20 dB without AGC (4,600 steps)", va["vad_20db_no_agc"]["multi_domain_eval"]))

    if "paper_faithful_presets_smoke_test" in st:
        pf = st["paper_faithful_presets_smoke_test"]
        out.append(row("7. `icassp2018_ge2e_config(pure_paper_mode=True)` (3×768 LSTM + 256 proj, 4,600 steps)", pf["icassp2018_ge2e_pure_paper"]["multi_domain_eval"]))
        out.append(row("7. `interspeech2021_dr_vectors_config(pure_paper_mode=True)` (200+56 split, 3×256 LReLU, 4,600 steps)", pf["interspeech2021_dr_vectors_pure_paper"]["multi_domain_eval"]))
        out.append(row("7. `odyssey2022_conformer_atp_config(pure_paper_mode=True)` (conv-before-MHSA, sin PE, ×2 subsample, 4,600 steps)", pf["odyssey2022_conformer_atp_pure_paper"]["multi_domain_eval"]))
        out.append(row("7. `odyssey2022_pfas_config(pure_paper_mode=True)` (`atp_ffn`, 32 tied keys × (16+48), 4,600 steps)", pf["odyssey2022_pfas_pure_paper"]["multi_domain_eval"]))
    out.append("")

    # Multi-seed stability
    out.append("### Multi-seed stability at converged `S_safe` (`r2_seed_and_control_results.json`)\n")
    out.append(
        "| Model (`Steps`) | Seeds | Pooled 1-utt EER per seed (%) | Pooled 1-utt mean ± sd | "
        "Macro 1-utt mean ± sd | LS 1-utt mean ± sd | Pooled 3-utt mean ± sd | Released seed-42 pooled / macro 1-utt |"
    )
    out.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    ms_all = r2.get("multi_seed_stability", {})
    for mid, mv in ms_all.items():
        s1, s3 = mv["summary_1utt"], mv["summary_3utt"]
        per = sorted(mv["per_seed"], key=lambda r: int(r["seed"]))
        seeds_str = ", ".join(str(r["seed"]) for r in per)
        vals_str = " / ".join(f2(r["summary"]["pooled_1utt_eer"]) for r in per)
        rel = ex[mid]
        out.append(
            f"| `{mid}` (`{rel['num_steps']:,}`) | {seeds_str} | {vals_str} | "
            f"{f2(s1['universal_pooled']['mean'])} ± {f2(s1['universal_pooled']['std'])} | "
            f"{f2(s1['macro_avg']['mean'])} ± {f2(s1['macro_avg']['std'])} | "
            f"{f2(s1['librispeech']['mean'])} ± {f2(s1['librispeech']['std'])} | "
            f"{f2(s3['universal_pooled']['mean'])} ± {f2(s3['universal_pooled']['std'])} | "
            f"{f2(d1(rel, 'universal_pooled')['eer_percent'])} / {f2(mac(rel, 1)['eer_percent'])} |"
        )
    out.append("")

    # Matched-data control
    md_ctrl = r2.get("matched_data_control", {})
    out.append("### Matched-data control (`r2_seed_and_control_results.json`)\n")
    out.append("| Training configuration | Steps | Speakers / utterances | LS 1-utt | CN 1-utt | VOX 1-utt | Macro 1-utt | Pooled 1-utt [95% CI] | Macro 3-utt | Pooled 3-utt |")
    out.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    def mrow_entry(label: str, steps: int, spk: int, utt: int, mde: dict) -> str:
        p1 = mde["domains_1utt"]["universal_pooled"]
        return (
            f"| {label} | `{steps:,}` | {spk} / {utt:,} | "
            f"{f2(mde['domains_1utt']['librispeech']['eer_percent'])} | "
            f"{f2(mde['domains_1utt']['cnceleb2']['eer_percent'])} | "
            f"{f2(mde['domains_1utt']['voxceleb1']['eer_percent'])} | "
            f"{f2(mde['universal_macro_avg_1utt']['eer_percent'])} | "
            f"{f2(p1['eer_percent'])} [{f2(p1['eer_ci95_low_percent'])}, {f2(p1['eer_ci95_high_percent'])}] | "
            f"{f2(mde['universal_macro_avg_3utt']['eer_percent'])} | "
            f"{f2(mde['domains_3utt']['universal_pooled']['eer_percent'])} |"
        )

    out.append(mrow_entry("`ls_conformer_small_ge2e` (LibriSpeech-only specialist) §", 2400, 232, 5788, ex["ls_conformer_small_ge2e"]["multi_domain_eval"]))
    if "univ_subsampled_5788_2400steps" in md_ctrl:
        m24 = md_ctrl["univ_subsampled_5788_2400steps"]
        out.append(mrow_entry("Universal subsampled to LibriSpeech utterance count (matched utterances & compute)", 2400, m24["num_train_speakers"], m24["num_train_utterances"], m24["multi_domain_eval"]))
    if "univ_subsampled_5788_4600steps" in md_ctrl:
        m46 = md_ctrl["univ_subsampled_5788_4600steps"]
        out.append(mrow_entry("Universal subsampled to LibriSpeech utterance count (`S_safe` 4,600-step schedule)", 4600, m46["num_train_speakers"], m46["num_train_utterances"], m46["multi_domain_eval"]))
    out.append(mrow_entry("`univ_conformer_small_ge2e` (full Universal 3-domain mix) §", 4600, 381, 9447, ex["univ_conformer_small_ge2e"]["multi_domain_eval"]))
    out.append("")

    # Step-0 fingerprint
    ck = fps["checkpoints"]
    hist = fps["histogram"]
    out.append(
        f"### Step-0 loss fingerprint of all {len(ck)} released checkpoints "
        f"(`step0_fingerprints.json`, code `{fps.get('code_commit', '')}`; tolerance {fps['tolerance']})\n"
    )
    out.append("Histogram: " + ", ".join(f"`{k}` = {v}" for k, v in sorted(hist.items())) + ".\n")
    out.append("| Checkpoint | Loss | Released `initial_loss` | `HEAD`, fixed T = 180 | `HEAD`, T ~ U(90,180) | Match |")
    out.append("| :--- | :--- | :---: | :---: | :---: | :--- |")
    for mid, v in sorted(ck.items(), key=lambda kv: (kv[1]["match"], kv[0])):
        fx = v["head_fixed_legacy_lambda"] if v.get("legacy_lambda_used") else v["head_fixed"]
        vr = v["head_variable_legacy_lambda"] if v.get("legacy_lambda_used") else v["head_variable"]
        out.append(f"| `{mid}` | `{v['loss_type']}` | {v['released_initial_loss']:.4f} | {fx:.4f} | {vr:.4f} | `{v['match']}` |")
    out.append("")

    # TFLite 180-frame verification EER & latency table (8 models = 16 FP32/INT8 binaries)
    out.append("### TFLite 180-frame (1.8 s) verification EER & latency (`benchmark_results.json`)\n")
    out.append(
        "| Model | FP32 / INT8 size (MB) | Ratio | Primary eval set | TFLite FP32 EER (%) [95% CI] | "
        "TFLite INT8 EER (%) [95% CI] | JAX full-seq (`<=500f`) EER (%) | `cos(FP32, INT8)` mean (min) | "
        "CPU latency FP32 / INT8 (ms, 50-run mean) |"
    )
    out.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    tfl_ex = bench.get("tflite_exports", {})
    for mid in (
        "univ_full_conformer_large_arcface",
        "univ_full_ecapa_c1024_arcface",
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
        if "domains_1utt_fp32" in te:
            prim = te["primary_domain"]
            r32 = te["domains_1utt_fp32"][prim]
            r8 = te["domains_1utt_int8"][prim]
            fp_ci = f" [{f2(r32['eer_ci95_low'])}, {f2(r32['eer_ci95_high'])}]"
            in_ci = f" [{f2(r8['eer_ci95_low'])}, {f2(r8['eer_ci95_high'])}]"
            cos_min_s = f" (`{te['cos_fp32_int8_min']:.6f}`)"
        else:
            prim = "voxceleb1_o (37,611 trials)"
            fp_ci = ""
            in_ci = ""
            cos_min_s = ""
        out.append(
            f"| [`wq2012/{mid}`](https://huggingface.co/wq2012/{mid}) | {fp_b / 1e6:.2f} / **{in_b / 1e6:.2f}** | "
            f"**`{fp_b / in_b:.2f}×`** | `{prim}` | "
            f"{f2(te['eer_percent'])}%{fp_ci} | "
            f"**{f2(te['eer_percent_int8'])}%**{in_ci} | "
            f"{f2(te['jax_full_seq_eer_percent'])}% | "
            f"**`{te['cos_fp32_int8_mean']:.6f}`**{cos_min_s} | "
            f"{f2(te['latency_ms_fp32_mean_50runs'])} / {f2(te['latency_ms_int8_mean_50runs'])} ms |"
        )
    out.append("")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", choices=["all", "flagship", "readme", "studies", "controls"], default="all")
    args = parser.parse_args()

    bench = load("benchmark_results.json")
    abl = load("ablation_results.json")
    vox = load("vox1_o_10spk_subset_results.json")
    r2 = load("r2_seed_and_control_results.json")
    fps = load("step0_fingerprints.json")
    flagship = load("full_data_flagship_results.json")
    pfas_sweep = load("pfas_hparam_sweep.json")
    ex = bench["experiments"]

    chunks: list[str] = []
    if args.section in ("all", "flagship"):
        chunks.extend(render_flagship_tables(flagship, pfas_sweep))
    if args.section in ("all", "readme"):
        chunks.extend(render_readme_tables(ex))
    if args.section in ("all", "studies"):
        chunks.extend(render_report_studies(ex))
    if args.section in ("all", "controls"):
        chunks.extend(render_section10_and_11(bench, abl, vox, r2, fps))

    sys.stdout.write("\n".join(chunks))


if __name__ == "__main__":
    main()
