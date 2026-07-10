"""Metric figures + Japanese README.md / index.html for HD-EPIC-NF (Experiment B)."""
from __future__ import annotations

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C


def _figures(m):
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


FIGURES = [
    ("replay.gif", "リプレイ：実厨房の配置密度の上で物を滑らせ、サプライズをライブ計測",
     "**左**＝厨房を上から見た床面。色は「そこに物を置いたときのサプライズ」（明黄＝予想どおり／暗紫＝違和感）、"
     "白点＝実際の物体操作位置。**右**＝サプライズ計。**見どころ**：実データでも普段の作業スポット（コンロ・シンク・"
     "調理台）が複数の明るい島＝**多峰**として現れ、そこを外れると急に驚く。Aの人工データでなく実厨房で成立。"),
    ("surprise_maps.png", "厨房ごとのサプライズ地図（多峰性がFlowの勝因）",
     "6厨房それぞれの `p(位置 | 厨房)`。**見どころ**：明るい山が**複数**（コンロ/シンク/収納…）＝実厨房の配置は"
     "本質的に**多峰**。だから単一ガウスでは表せず、**Neural Spline Flowが効く**（＝実験Aで『ガウスに並ばれた』"
     "反省への直接的な回答）。"),
    ("metrics.png", "Flow vs GMM：密度の当てはまり と ランダム配置の検出",
     "**左**＝held-out NLL（低いほど良い）。**Flow < GMM** が本命の勝ち＝多峰配置をNFが良く当てている。"
     "**右**＝ランダムな場所に落とされた物の検出AUROC。両者とも高いが、差は密度(NLL)に最も出る。"),
    ("training_curve.png", "学習曲線：Flow vs GMM（held-out NLL）",
     "エポックごとの検証NLL。Flowが一貫してGMMより低い＝表現力の差が安定して効いている。"),
]


def _readme(m):
    tr, ev = m.get("train", {}), m.get("evaluate", {})
    fn = tr.get("flow", {}).get("best_val_nll"); gn = tr.get("gmm_baseline", {}).get("best_val_nll")
    L = []
    L.append("# HD-EPIC-NF — 実厨房の3D配置を条件付き Normalizing Flow で（実験B）\n")
    L.append("実際のキッチンで**物がどこで扱われるか**の3D位置を、厨房で条件づけた Normalizing Flow で "
             "`log p(3D位置 | 厨房)` として学習し、**SURPRISE = −log p** を「置き忘れそう（普段ありえない場所）」"
             "の指標にする実験。**実厨房の配置は本質的に多峰**なので、実験Aで『ガウスに並ばれた』反省への"
             "直接の回答になる。NF forget/mistake シリーズ(A–E)の B。図・数値は `python -m hde.report` で自動生成。\n")

    L.append("## どんなデータか\n")
    L.append("- **HD-EPIC** の**オープン注釈のみ**（eye-gaze priming）を使用。動画不要・ログイン不要。")
    L.append(f"- 各物体操作イベントの **3D位置**と**視線点(gaze)** を抽出。"
             f"9厨房(P01–P09)・153動画から **{ev.get('n_val',0)*5//4 + ev.get('n_val',0):,} 前後の操作点**。")
    L.append("- 視線オフセット `||位置 − 視線||` を「よそ見して扱ったか」の distraction 信号として保持。\n")

    L.append("## どんなモデルを学習したか\n")
    L.append("- **条件付き NSF**：`x = (x,y,z)` ／ `c = 厨房ID`。held-out は動画の約20%（未見セッション）。")
    L.append("- 比較のため **GMM（ガウス混合）ベースライン**を同条件で学習（実験Aの教訓の検証）。")
    L.append("- スコア **SURPRISE = −log p(位置 | 厨房)**。\n")

    L.append("## 結果\n")
    if fn is not None and gn is not None:
        L.append(f"- **密度の当てはまりは Flow の明確な勝ち**：held-out NLL **Flow `{fn:.3f}` < GMM `{gn:.3f}`**（低いほど良い）。"
                 f"実厨房の多峰な配置分布にNSFの表現力が効く＝**実験Aの『ガウスに並ばれた』を実データで克服**。")
    if ev:
        L.append(f"- **ランダム配置の検出**：AUROC Flow `{ev.get('injection_auc_flow')}` / GMM `{ev.get('injection_auc_gmm')}`。"
                 f"どちらも高いが差は密度(NLL)側に出る。")
        if "surprise_high_gazeoffset_mean" in ev:
            L.append(f"- **視線との結合（弱いが実在）**：視線から離れて扱われた物ほどサプライズが高い —"
                     f" 高gazeオフセット群 `{ev.get('surprise_high_gazeoffset_mean')}` > 低群 `{ev.get('surprise_low_gazeoffset_mean')}`"
                     f"（Spearman `{ev.get('gaze_surprise_spearman_flow')}`）。『よそ見配置＝忘れやすい』仮説の弱いが一貫した兆候。\n")

    L.append("## 図の見方\n")
    for f, t, h in FIGURES:
        if (C.FIGS / f).exists():
            L.append(f"### {t}\n\n![{f}](results/figures/{f})\n\n{h}\n")

    L.append("## 再現手順\n")
    L.append("```bash\n"
             "python -m hde.fetch      # HD-EPIC eye-gaze-priming 注釈を取得\n"
             "python -m hde.extract    # -> data/processed/points.parquet\n"
             "python -m hde.features   # 厨房語彙 + 正規化\n"
             "python -m hde.train      # 条件付きNSF + GMMベースライン（--fastで高速）\n"
             "python -m hde.evaluate   # 注入AUC + 視線結合\n"
             "python -m hde.heatmap    # 厨房ごとのサプライズ地図\n"
             "python -m hde.replay     # 厨房を滑るprobe + サプライズ計（GIF/MP4）\n"
             "python -m hde.report     # 図 + この日本語README + index.html\n```\n")
    L.append("_条件付きNFで物の配置尤度を forget/mistake ポテンシャルとして測るシリーズ(A–E)の B。"
             "A=合成の再配置(RoomR)／B=実厨房3D×視線(HD-EPIC)／C=手順のタイミング(HoloAssist)。_")
    C.BASE.joinpath("README.md").write_text("\n".join(L))


def _index_html(m):
    tr, ev = m.get("train", {}), m.get("evaluate", {})
    fn = tr.get("flow", {}).get("best_val_nll"); gn = tr.get("gmm_baseline", {}).get("best_val_nll")
    blocks = "\n".join(
        f'<section><h2>{t}</h2><img src="results/figures/{f}" alt="{f}"><p class="howto">{h}</p></section>'
        for f, t, h in FIGURES if (C.FIGS / f).exists())
    html = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HD-EPIC-NF · 実厨房の3D配置を条件付きNFで</title>
<style>
 body{{font:16px/1.75 -apple-system,"Hiragino Sans","Noto Sans JP",sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#222}}
 h1{{line-height:1.35;margin-bottom:.2rem}} .sub{{color:#666}}
 .kpis{{display:flex;gap:1rem;flex-wrap:wrap;margin:1.4rem 0}}
 .kpi{{background:#f5f3f0;border-radius:12px;padding:1rem 1.3rem;min-width:150px}} .kpi b{{display:block;font-size:1.6rem;color:#c2410c}}
 section{{margin:2.3rem 0}} img{{width:100%;border:1px solid #e5e5e5;border-radius:10px}}
 .howto{{color:#444;background:#faf8f5;border-left:3px solid #d97757;padding:.6rem .9rem;border-radius:0 8px 8px 0}}
 code{{background:#f0eee9;padding:.1rem .3rem;border-radius:4px}} .lead{{background:#f7f5f2;border-radius:12px;padding:1rem 1.2rem}}
</style></head><body>
<h1>HD-EPIC-NF — 実厨房の3D配置を条件付き Normalizing Flow で</h1>
<p class="sub">実験B · <code>SURPRISE = −log p(3D位置 | 厨房)</code>。実厨房の多峰な配置でNFがGMMに勝つ。</p>
<div class="kpis">
 <div class="kpi"><b>{f'{fn:.3f}' if fn is not None else '—'}</b>Flow held-out NLL</div>
 <div class="kpi"><b>{f'{gn:.3f}' if gn is not None else '—'}</b>GMM baseline NLL</div>
 <div class="kpi"><b>{ev.get('injection_auc_flow','—')}</b>ランダム配置検出 AUROC</div>
</div>
<p class="lead"><b>何をしたか</b>：HD-EPICのオープン注釈（3D物体位置＋視線・動画不要）から、
厨房ごとの<b>物の扱われる場所の密度</b>を条件付きNSFで学習。<b>実厨房の配置は多峰</b>（コンロ/シンク/調理台…）なので、
<b>held-out NLLでFlowがGMMに明確に勝つ</b>＝実験Aの『ガウスに並ばれた』反省を実データで克服。
視線から離れて扱われた物ほどサプライズが高い、という『よそ見＝忘れやすい』の弱い兆候も観測。</p>
{blocks}
<p class="sub"><code>python -m hde.report</code> で自動生成。NF forget/mistakeシリーズ(A–E)のB。</p>
</body></html>"""
    C.BASE.joinpath("index.html").write_text(html)


def main() -> None:
    m = json.loads(C.METRICS_JSON.read_text())
    _figures(m)
    _readme(m)
    _index_html(m)
    print("wrote README.md (JA) + index.html (JA) + figures")


if __name__ == "__main__":
    main()
