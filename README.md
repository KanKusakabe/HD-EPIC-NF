# HD-EPIC-NF — 実厨房の3D配置を条件付き Normalizing Flow で（実験B）

実際のキッチンで**物がどこで扱われるか**の3D位置を、厨房で条件づけた Normalizing Flow で `log p(3D位置 | 厨房)` として学習し、**SURPRISE = −log p** を「置き忘れそう（普段ありえない場所）」の指標にする実験。**実厨房の配置は本質的に多峰**なので、実験Aで『ガウスに並ばれた』反省への直接の回答になる。NF forget/mistake シリーズ(A–E)の B。図・数値は `python -m hde.report` で自動生成。

## どんなデータか

- **HD-EPIC** の**オープン注釈のみ**（eye-gaze priming）を使用。動画不要・ログイン不要。
- 各物体操作イベントの **3D位置**と**視線点(gaze)** を抽出。9厨房(P01–P09)・153動画から **13,950 前後の操作点**。
- 視線オフセット `||位置 − 視線||` を「よそ見して扱ったか」の distraction 信号として保持。

## どんなモデルを学習したか

- **条件付き NSF**：`x = (x,y,z)` ／ `c = 厨房ID`。held-out は動画の約20%（未見セッション）。
- 比較のため **GMM（ガウス混合）ベースライン**を同条件で学習（実験Aの教訓の検証）。
- スコア **SURPRISE = −log p(位置 | 厨房)**。

## 結果

- **密度の当てはまりは Flow の明確な勝ち**：held-out NLL **Flow `1.030` < GMM `1.253`**（低いほど良い）。実厨房の多峰な配置分布にNSFの表現力が効く＝**実験Aの『ガウスに並ばれた』を実データで克服**。
- **ランダム配置の検出**：AUROC Flow `0.9825` / GMM `0.9769`。どちらも高いが差は密度(NLL)側に出る。
- **視線との結合（弱いが実在）**：視線から離れて扱われた物ほどサプライズが高い — 高gazeオフセット群 `1.65` > 低群 `1.408`（Spearman `0.066`）。『よそ見配置＝忘れやすい』仮説の弱いが一貫した兆候。

## 図の見方

### リプレイ：実厨房の配置密度の上で物を滑らせ、サプライズをライブ計測

![replay.gif](results/figures/replay.gif)

**左**＝厨房を上から見た床面。色は「そこに物を置いたときのサプライズ」（明黄＝予想どおり／暗紫＝違和感）、白点＝実際の物体操作位置。**右**＝サプライズ計。**見どころ**：実データでも普段の作業スポット（コンロ・シンク・調理台）が複数の明るい島＝**多峰**として現れ、そこを外れると急に驚く。Aの人工データでなく実厨房で成立。

### 厨房ごとのサプライズ地図（多峰性がFlowの勝因）

![surprise_maps.png](results/figures/surprise_maps.png)

6厨房それぞれの `p(位置 | 厨房)`。**見どころ**：明るい山が**複数**（コンロ/シンク/収納…）＝実厨房の配置は本質的に**多峰**。だから単一ガウスでは表せず、**Neural Spline Flowが効く**（＝実験Aで『ガウスに並ばれた』反省への直接的な回答）。

### Flow vs GMM：密度の当てはまり と ランダム配置の検出

![metrics.png](results/figures/metrics.png)

**左**＝held-out NLL（低いほど良い）。**Flow < GMM** が本命の勝ち＝多峰配置をNFが良く当てている。**右**＝ランダムな場所に落とされた物の検出AUROC。両者とも高いが、差は密度(NLL)に最も出る。

### 学習曲線：Flow vs GMM（held-out NLL）

![training_curve.png](results/figures/training_curve.png)

エポックごとの検証NLL。Flowが一貫してGMMより低い＝表現力の差が安定して効いている。

## 再現手順

```bash
python -m hde.fetch      # HD-EPIC eye-gaze-priming 注釈を取得
python -m hde.extract    # -> data/processed/points.parquet
python -m hde.features   # 厨房語彙 + 正規化
python -m hde.train      # 条件付きNSF + GMMベースライン（--fastで高速）
python -m hde.evaluate   # 注入AUC + 視線結合
python -m hde.heatmap    # 厨房ごとのサプライズ地図
python -m hde.replay     # 厨房を滑るprobe + サプライズ計（GIF/MP4）
python -m hde.report     # 図 + この日本語README + index.html
```

_条件付きNFで物の配置尤度を forget/mistake ポテンシャルとして測るシリーズ(A–E)の B。A=合成の再配置(RoomR)／B=実厨房3D×視線(HD-EPIC)／C=手順のタイミング(HoloAssist)。_