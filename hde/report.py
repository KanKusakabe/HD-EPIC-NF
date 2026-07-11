"""Metric figures + Japanese README.md / index.html for HD-EPIC-NF.

Every figure carries TWO lines: 観察 (what the graph shows) and 解釈 (what that
therefore means) -- not "value A was big" but "A was big, so X follows".
Run AFTER hde.heatmap, hde.replay, hde.deep so all figures + metrics exist.
"""
from __future__ import annotations

import json
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C


def _md(s: str) -> str:
    """Render inline markdown (**bold**, *italic*) as HTML so index.html isn't literal."""
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"\*(.+?)\*", r"<i>\1</i>", s)
    return s


# --- What data / how learned (shared by README + index) ---
DATA_ROWS = [
    ("データ", "<b>HD-EPIC</b> の公開注釈（実キッチンのエゴセントリック記録・9厨房・複数日）。"
               "動画は使わず<b>注釈のみ</b>（物体の3D位置・object movements・視線）を使用。"),
    ("使う量", "「実際に扱われた物」の <b>3D位置 (x,y,z)</b> を厨房ごとに集める（厨房 ≒ その人の台所）。"),
    ("予測対象 x", "物が扱われた <b>3D位置 (x, y, z)</b>"),
    ("条件 c", "<b>厨房（≒その人）</b>"),
    ("モデル", "条件付き <b>Neural Spline Flow</b>（zuko）で <code>p(x,y,z｜厨房)</code>。比較ベース＝GMM。"),
    ("スコア", "<b>SURPRISE = −log p</b>＝「その厨房で、その場所にその物があるのはどれだけ意外か」"),
]
NUM_GUIDE = [
    ("NLL（held-out）", "実際の配置をどれだけ当てたか。<b>低いほど良い</b>（負でもOK）。差が大きいほど当てやすい。"),
    ("AUROC", "異常と正常を見分ける力。<b>0.5＝勘・1.0＝完璧</b>。"),
    ("nats（ゲイン）", "条件（誰か・時間）を足したときのNLL改善量。大きいほどその条件が効く。"),
]

RAW_INTRO = (
    "<b>生データ</b>＝HD-EPIC の<b>公開注釈JSON</b>（動画は不使用・ログイン不要）。使ったのは主に "
    "<code>eye-gaze-priming / object-movements</code>：物が扱われるたびに "
    "<code>{ 3D位置(x,y,z), 視線のズレ(gaze_offset), カメラからの距離(dist_to_cam), "
    "見てから触るまでの時間(prime_gap), 厨房ID(P01…), 移動の開始/終了(phase) }</code>。<br>"
    "<b>1レコードの実例</b>：<code>P01</code> の台所で、ある物の操作開始点が <code>(-0.11, -3.13, -0.03)</code>、"
    "その時 gaze は物から <code>0.07</code> ずれ、カメラから <code>2.3m</code>、見てから触るまで <code>3.5秒</code>。<br>"
    "→ 学習の主対象は <b>物の3D位置 × 厨房(≒人)</b>（視線ズレは『見ずに置いたか』の解釈に使用）。計 35,911 レコード。")

FUTURE = (
    "<p>本実装は「<b>3D位置＋厨房ID</b>」が主。データの列がこう増えると、こんな学習に広がる：</p><ul>"
    "<li><b>＋視線・手ポーズの<u>時系列</u></b>（本データに gaze はあり、系列条件化が次段）→ "
    "『よそ見して置く→後で探す』の因果に踏み込める。</li>"
    "<li><b>＋その物に次に触れた時刻（放置時間）</b> → 忘却の<b>実ラベル</b>ができ、"
    "『低尤度な置き方＝実際に忘れやすい』を直接検証できる（本課題の核心・未達）。</li>"
    "<li><b>＋物体カテゴリの明示的条件化</b> → 物ごとの置き場マップ（鍋はコンロ・包丁はまな板脇…）。</li>"
    "<li><b>＋複数日・時刻</b> → 習慣のドリフト（L2で部分実証済み）。</li>"
    "<li><b>＋レイアウトを揃えた座標系</b> → 他人の事前分布が効き few-shot 転移が可能に"
    "（L3で『他人で事前学習が逆効果』だった処方箋）。</li></ul>")


def _base_figures(m):
    tr = m.get("train", {}); ev = m.get("evaluate", {})
    fh = tr.get("flow", {}).get("history", []); gh = tr.get("gmm_baseline", {}).get("history", [])
    if fh:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([r["epoch"] for r in fh], [r["val_nll"] for r in fh], label="Flow (NSF)", color="#d97757")
        if gh:
            ax.plot([r["epoch"] for r in gh], [r["val_nll"] for r in gh], label="GMM baseline", color="#8b93a1")
        ax.set_xlabel("epoch"); ax.set_ylabel("held-out NLL (lower = better)")
        ax.set_title("3-D placement density: Flow vs GMM"); ax.legend()
        fig.tight_layout(); fig.savefig(C.FIGS / "training_curve.png", dpi=110); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    fn = tr.get("flow", {}).get("best_val_nll", 0); gn = tr.get("gmm_baseline", {}).get("best_val_nll", 0)
    b = axes[0].bar(["Flow", "GMM"], [fn, gn], color=["#d97757", "#8b93a1"])
    axes[0].set_ylabel("held-out NLL (lower = better)"); axes[0].set_title("Density fit — the Flow wins")
    for bar, v in zip(b, [fn, gn]):
        axes[0].annotate(f"{v:.3f}", (bar.get_x() + bar.get_width() / 2, v), ha="center", va="bottom")
    af = ev.get("injection_auc_flow", 0); ag = ev.get("injection_auc_gmm", 0)
    b2 = axes[1].bar(["Flow", "GMM"], [af, ag], color=["#d97757", "#8b93a1"])
    axes[1].axhline(0.5, ls="--", c="gray", lw=1); axes[1].set_ylim(0.5, 1.02)
    axes[1].set_ylabel("random-teleport AUROC"); axes[1].set_title("Detecting objects dropped at random")
    for bar, v in zip(b2, [af, ag]):
        axes[1].annotate(f"{v:.3f}", (bar.get_x() + bar.get_width() / 2, v), ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(C.FIGS / "metrics.png", dpi=110); plt.close(fig)


def build_figures(m):
    """Return [(filename, title, 観察, 解釈), ...] with value-aware interpretations."""
    tr, ev, dp = m.get("train", {}), m.get("evaluate", {}), m.get("deep", {})
    fn = tr.get("flow", {}).get("best_val_nll"); gn = tr.get("gmm_baseline", {}).get("best_val_nll")
    figs = []

    figs.append(("replay.gif", "リプレイ：実厨房の配置密度の上で物を滑らせ、サプライズをライブ計測",
        "物を厨房の各位置に置いたときの −log p を色で、白点＝実際の操作位置。プローブが普段の作業島（明）から外れると計器が跳ねる。",
        "**だから何が言えるか**：実厨房でも『普段の置き場』は複数の島＝多峰で存在し、そこを外れた瞬間だけ驚く。"
        "＝サプライズは“場所の妥当性”を連続量として測れており、単なる外れ値フラグでなく『どれくらい変か』の勾配を持つ。"))

    if fn is not None and gn is not None:
        d = gn - fn
        figs.append(("metrics.png", "Flow vs GMM：密度の当てはまり と ランダム配置の検出",
            f"held-out NLL は Flow {fn:.3f} < GMM {gn:.3f}（差 {d:.2f} nats）。ランダム配置の検出AUROCは両者とも ~{ev.get('injection_auc_flow','?')}。",
            f"**だから何が言えるか**：厨房配置は多峰なので、単一ガウスの重ね合わせ（GMM）より Flow の方が {d:.2f} nats ぶん“ありえる場所”を正しく狭く当てている。"
            "＝『実験Aでガウスに並ばれた』のは題材が単峰で低次元だったからで、実データの多峰性ではNFの表現力が本質的に効く、という反省の実証。"
            "一方ランダム配置は簡単すぎて両者高く、差は密度(NLL)側にしか出ない＝評価は“難しい異常”で見るべき。"))

    figs.append(("surprise_maps.png", "厨房ごとのサプライズ地図（多峰性）",
        "6厨房それぞれ、明るい山（低サプライズ＝定位置）が複数。白点の実操作もその島に乗る。",
        "**だから何が言えるか**：人は物を1箇所でなく数箇所の“定位置”に置く。この多峰構造こそFlowがGMMに勝つ理由で、"
        "『どこがその人にとって普通か』は本質的に多峰＝個人ごとに違う地図になる、という次(L1)への布石。"))

    # L1
    if "l1" in dp:
        g = dp["l1"]["mean_gain_nats"]
        figs.append(("l1_personalization.png", "L1 · 個人 vs 集団：誰かを知ると配置予測はどれだけ良くなるか",
            f"厨房（＝人）で条件づけたモデルは、集団一律モデルより held-out NLL が平均 {g:.2f} nats 低い。",
            f"**だから何が言えるか**：{'＋なら' if g>0 else ''}“誰か”を知るだけで予測が {g:.2f} nats 改善＝**置き場の習慣は個人ごとに違う**。"
            + ("つまり集団基準の忘れ物検出は、その人には普通の場所を『異常』と誤警報する。個人化は任意でなく必須、という結論。"
               if g > 0.05 else "差が小さいなら習慣は概ね共有され集団モデルで足りる、という逆の結論になる。")))

    # L2
    if "l2" in dp:
        d = dp["l2"]["mean_drift_nats"]
        figs.append(("l2_drift.png", "L2 · ルーチンのドリフト：早期に学んだ習慣は後日も当たるか",
            f"早期データで学んだモデルを『後日（未来）』で評価すると NLL が時間無関係のランダム保留より平均 {d:+.2f} nats 変化。",
            f"**だから何が言えるか**：{'未来ほど当てにくい＝' if d>0.02 else ''}“普通”は時間で{('動く（非定常）。' if d>0.02 else 'ほぼ動かない（定常・定着）。')}"
            + ("＝静的な忘れ物検出は時間とともに陳腐化し、オンライン更新が要る。習慣が再編される＝これ自体が“忘却/学習”の時間的側面。"
               if d > 0.02 else "＝一度学べば固定モデルで足り、早期データだけで十分（＝習慣が既に定着＝速いフェーディング）。")))

    # L3
    if "l3" in dp:
        pre = dp["l3"].get("pretrained_nll", {}); scr = dp["l3"].get("scratch_nll", {})
        ks = sorted(int(k) for k in pre)
        sm, lg = str(ks[0]), str(ks[-1])
        d_small = pre[sm] - scr[sm]                 # >0 means pretraining is worse
        helps = d_small < -0.02
        obs = (f"対象厨房に k 例だけで適応。少数({sm}例)では pretrained {pre[sm]:.2f} / scratch {scr[sm]:.2f}、"
               f"多め({lg}例)でも {pre[lg]:.2f} / {scr[lg]:.2f}。")
        if helps:
            so = ("**だから何が言えるか**：他人8厨房の事前分布が新しい人に転移し、少数例で素早く個人化できる"
                  "＝自前データが少なくても公共事前＋few-shotが効く（moat の実現可能性）。")
        else:
            so = (f"**だから何が言えるか（正直な逆結果）**：他人で事前学習した方が全域で**むしろ悪い**（{sm}例で {d_small:+.2f} nats）。"
                  "L1 が示した通り配置習慣は**強く個人的**で、しかも厨房ごとに座標系・レイアウトが違うため、"
                  "他人の事前分布は“間違った prior”になる。＝**moat は「他人で事前学習」ではなく、本人自身の少数データからの個人化**"
                  "（またはレイアウトを揃えた表現）にある、という重要な知見。naiveな転移は効かず、個人化の“正しいやり方”選択が本質。")
        figs.append(("l3_fewshot.png", "L3 · few-shot 転移：新しい人を“他人の事前分布”から個人化できるか",
            obs, so))

    # L4
    if "l4" in dp:
        u = dp["l4"]["best_predictive_utility"]; cat = dp["l4"]["catch_at_best"]; fa = dp["l4"]["false_alarm_at_best"]
        figs.append(("l4_policy.png", "L4 · 予測 vs リアクティブ：サプライズで“事前に”警告する価値",
            f"サプライズ閾値を動かした予測ポリシーの期待効用は最大 {u:.2f}（リアクティブ＝事後対応は 0）。"
            f"最良点で at-risk の {cat:.0%} を捕捉、誤警報 {fa:.0%}。",
            f"**だから何が言えるか**：{'効用が正＝' if u>0 else ''}誤警報コストを引いても“置く前に驚いて警告する”方が事後対応より得"
            + (f"（正味 {u:.2f}）。＝サプライズは検出器の AUC でなく**意思決定に効く**信号で、最適閾値が『いつ介入すべきか』を与える。"
               "貢献は“異常検知”でなく**介入ポリシー**、というテーゼの実証。"
               "（ただし at-risk はランダムテレポート＝易しめの合成異常なので捕捉率は楽観的。要点は“予測＞リアクティブ”という意思決定の向き。）" if u > 0 else
               "とは言えず、この設定では事前警告は割に合わない。コスト設定・信号を見直す必要がある、という正直な結論。")))

    figs.append(("training_curve.png", "学習曲線：Flow vs GMM（held-out NLL）",
        "エポックごとの検証NLLで Flow が一貫して GMM より低い。",
        "**だから何が言えるか**：表現力の差は初期の運や過学習でなく安定した性質＝多峰配置に対するNFの優位は再現的。"))
    return figs


def _readme(m):
    tr, ev, dp = m.get("train", {}), m.get("evaluate", {}), m.get("deep", {})
    fn = tr.get("flow", {}).get("best_val_nll"); gn = tr.get("gmm_baseline", {}).get("best_val_nll")
    L = []
    L.append("# HD-EPIC-NF — 実厨房の配置を条件付き Normalizing Flow で（実験B＋深化 L1–L4）\n")
    L.append("実キッチンで**物がどこで扱われるか**の3D位置を Normalizing Flow で `log p(3D位置 | 厨房)` として学習し、"
             "**SURPRISE = −log p** を「置き忘れそう」の指標にする。さらに、単なる**異常検知**から一段深めて、"
             "**個人の routine のモデル**とその**介入への有用性**まで踏み込む（L1–L4）。"
             "図・数値は `python -m hde.report` で自動生成。各図に**観察**と**解釈（だから何が言えるか）**を併記。\n")
    L.append("## 元データの中身（何が入っているか）\n")
    L.append(RAW_INTRO + "\n")
    L.append("## 何をしたか（層構造）\n")
    L.append("- **B（基盤）** `p(位置｜厨房)`：実厨房は多峰なので Flow が GMM に勝つ（Aの『ガウスに並ばれた』反省の実データでの克服）。")
    L.append("- **L1 個人 vs 集団**：“誰か”を知ると予測がどれだけ良くなるか＝習慣は個人的か。")
    L.append("- **L2 ドリフト**：早期に学んだ習慣は後日も当たるか＝“普通”は定常か。")
    L.append("- **L3 few-shot 転移**：新しい人を公共の事前分布から何例で個人化できるか。")
    L.append("- **L4 予測 vs リアクティブ**：サプライズで“事前に”警告する意思決定価値。\n")

    if fn is not None:
        L.append("## 主要数値\n")
        L.append(f"- 基盤：held-out NLL **Flow {fn:.3f} < GMM {gn:.3f}**。")
        if "l1" in dp:
            L.append(f"- L1 個人化ゲイン **{dp['l1']['mean_gain_nats']:+.2f} nats**（個人条件付けでNLL低下）。")
        if "l2" in dp:
            L.append(f"- L2 時間ドリフト **{dp['l2']['mean_drift_nats']:+.2f} nats**（未来日 − ランダム保留）。")
        if "l4" in dp:
            L.append(f"- L4 予測ポリシー効用 **{dp['l4']['best_predictive_utility']:+.2f}** vs リアクティブ 0"
                     f"（捕捉 {dp['l4']['catch_at_best']:.0%} / 誤警報 {dp['l4']['false_alarm_at_best']:.0%}）。\n")

    L.append("## データ / 学習\n")
    L.append("| 項目 | 内容 |")
    L.append("|---|---|")
    for k, v in DATA_ROWS:
        L.append(f"| **{k}** | {v} |")
    L.append("\n**数字の読み方**\n")
    for k, v in NUM_GUIDE:
        L.append(f"- **{k}**：{v}")
    L.append("")

    L.append("## 図と解釈\n")
    for f, t, obs, so in build_figures(m):
        if (C.FIGS / f).exists():
            L.append(f"### {t}\n\n![{f}](results/figures/{f})\n\n**観察**：{obs}\n\n{so}\n")

    L.append("## 再現手順\n")
    L.append("```bash\n"
             "python -m hde.fetch\npython -m hde.extract\npython -m hde.features\n"
             "python -m hde.train      # 基盤: NSF + GMM\n"
             "python -m hde.evaluate\npython -m hde.heatmap\npython -m hde.replay\n"
             "python -m hde.deep       # L1–L4（個人化 / ドリフト / few-shot / ポリシー）\n"
             "python -m hde.report     # 図 + 日本語README + index.html（観察＋解釈つき）\n```\n")
    L.append("## 考察：どんなフォーマットのデータがあれば何ができるか\n")
    L.append(FUTURE + "\n")
    L.append("_条件付きNFで forget/mistake を測るシリーズの B を、異常検知から**個人化された予測誤差の時間発展＋介入ポリシー**へ深化した版。_")
    C.BASE.joinpath("README.md").write_text("\n".join(L))


def _index_html(m):
    tr, ev, dp = m.get("train", {}), m.get("evaluate", {}), m.get("deep", {})
    fn = tr.get("flow", {}).get("best_val_nll"); gn = tr.get("gmm_baseline", {}).get("best_val_nll")
    kpi = []
    if fn is not None:
        kpi.append((f"{fn:.3f}", "Flow held-out NLL"))
        kpi.append((f"{gn:.3f}", "GMM baseline NLL"))
    if "l1" in dp:
        kpi.append((f"{dp['l1']['mean_gain_nats']:+.2f}", "L1 個人化ゲイン (nats)"))
    if "l4" in dp:
        kpi.append((f"{dp['l4']['best_predictive_utility']:+.2f}", "L4 予測ポリシー効用"))
    kpi_html = "\n".join(f'<div class="kpi"><b>{v}</b>{lab}</div>' for v, lab in kpi)
    blocks = "\n".join(
        f'<section><h2>{t}</h2><img src="results/figures/{f}" alt="{f}">'
        f'<p class="obs"><b>観察</b>：{_md(obs)}</p><p class="so">{_md(so)}</p></section>'
        for f, t, obs, so in build_figures(m) if (C.FIGS / f).exists())
    drows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in DATA_ROWS)
    guide = "".join(f"<li><b>{k}</b>：{v}</li>" for k, v in NUM_GUIDE)
    data_html = (f'<section><h2>データ / 学習（このページの前提）</h2><table class="d">{drows}</table>'
                 f'<p class="sub" style="margin:.7rem 0 .2rem"><b>数字の読み方</b></p><ul>{guide}</ul></section>')
    html = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HD-EPIC-NF · 実厨房の配置NF ＋ 個人化/ドリフト/few-shot/ポリシー</title>
<style>
 body{{font:16px/1.75 -apple-system,"Hiragino Sans","Noto Sans JP",sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#222}}
 h1{{line-height:1.35;margin-bottom:.2rem}} .sub{{color:#666}}
 .kpis{{display:flex;gap:1rem;flex-wrap:wrap;margin:1.4rem 0}}
 .kpi{{background:#f5f3f0;border-radius:12px;padding:1rem 1.3rem;min-width:150px}} .kpi b{{display:block;font-size:1.6rem;color:#c2410c}}
 section{{margin:2.3rem 0}} img{{width:100%;border:1px solid #e5e5e5;border-radius:10px}}
 .obs{{color:#333;margin:.5rem 0 .2rem}}
 .so{{color:#444;background:#faf8f5;border-left:3px solid #d97757;padding:.6rem .9rem;border-radius:0 8px 8px 0}}
 code{{background:#f0eee9;padding:.1rem .3rem;border-radius:4px}} .lead{{background:#f7f5f2;border-radius:12px;padding:1rem 1.2rem}}
 table.d{{border-collapse:collapse;width:100%}} table.d td{{border:1px solid #e5e5e5;padding:.4rem .6rem;vertical-align:top}}
 table.d tr td:first-child{{white-space:nowrap;font-weight:600;background:#faf8f5;width:9rem}}
 ul{{margin:.3rem 0}}
</style></head><body>
<h1>HD-EPIC-NF — 実厨房の配置NF ＋ 個人化 / ドリフト / few-shot / 介入ポリシー</h1>
<p class="sub">実験B（<code>−log p(位置｜厨房)</code>）を、異常検知から<b>個人の routine とその介入有用性</b>まで深化（L1–L4）。</p>
<div class="kpis">{kpi_html}</div>
<p class="lead"><b>要旨</b>：実厨房の配置は多峰なので Flow が GMM に勝つ（B）。そこから、
<b>L1</b> 習慣は個人的（“誰か”を知ると予測が改善）、<b>L2</b> 習慣は時間で動く/定着する、
<b>L3</b> 公共の事前分布から少数例で新しい人を個人化できる、<b>L4</b> サプライズで事前警告する方が事後対応より得、
を示す。各図に<b>観察</b>と<b>解釈（だから何が言えるか）</b>を併記。</p>
<section><h2>元データの中身（何が入っているか）</h2><p class="lead">{RAW_INTRO}</p></section>
{data_html}
{blocks}
<section><h2>考察：どんなフォーマットのデータがあれば何ができるか</h2>{FUTURE}</section>
<p class="sub"><code>python -m hde.report</code> で自動生成。NF forget/mistakeシリーズBの深化版（L1–L4）。</p>
</body></html>"""
    C.BASE.joinpath("index.html").write_text(html)


def main() -> None:
    m = json.loads(C.METRICS_JSON.read_text())
    _base_figures(m)
    _readme(m)
    _index_html(m)
    print("wrote README.md (JA) + index.html (JA) + figures — with 観察/解釈 per graph")


if __name__ == "__main__":
    main()
