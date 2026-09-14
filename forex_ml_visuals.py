"""Standalone heatmaps and an offline interactive research report."""
from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from forex_ml_tournament import OUT, ROOT, build_features, load_hourly

LABEL = {"direction": "Yön tahmini", "high_volatility": "Yüksek oynaklık tahmini"}


def heatmap(frame, title, path, center=.5, percent=True, minimum=None, maximum=None):
    values = frame.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    span = max(float(np.max(np.abs(finite - center))), .01) if finite.size else .01
    vmin = center - span if minimum is None else minimum
    vmax = center + span if maximum is None else maximum
    fig, ax = plt.subplots(figsize=(max(9, len(frame.columns) * .85), max(4.5, len(frame) * .4)))
    fig.patch.set_facecolor("#101826")
    ax.set_facecolor("#101826")
    shown = values * 100 if percent else values
    im = ax.imshow(shown, cmap="RdYlGn", aspect="auto", vmin=vmin * (100 if percent else 1), vmax=vmax * (100 if percent else 1))
    ax.set_xticks(range(len(frame.columns)), frame.columns.astype(str), rotation=40, ha="right", color="white")
    ax.set_yticks(range(len(frame)), frame.index.astype(str), color="white")
    ax.set_title(title, color="white", pad=20, fontsize=14)
    if values.size <= 400:
        for row in range(len(frame)):
            for col in range(len(frame.columns)):
                value = shown[row, col]
                if np.isfinite(value):
                    ax.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=8, color="#102030")
    cbar = fig.colorbar(im, ax=ax, shrink=.75)
    cbar.ax.tick_params(colors="white")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    display = np.round(shown, 2)
    chart = go.Figure(go.Heatmap(z=display, x=frame.columns.astype(str), y=frame.index.astype(str),
        colorscale="RdYlGn", zmid=center * (100 if percent else 1),
        zmin=vmin * (100 if percent else 1), zmax=vmax * (100 if percent else 1),
        text=display, texttemplate="%{text}", hovertemplate="%{y}<br>%{x}: %{z}<extra></extra>"))
    chart.update_layout(title=title, template="plotly_dark", height=max(400, len(frame) * 29),
                        margin=dict(l=110, r=30, t=70, b=120))
    return chart


def generate(output=OUT):
    output = Path(output)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    dev = pd.read_csv(output / "development.csv")
    final = pd.read_csv(output / "final_metrics.csv")
    trades = pd.read_csv(output / "cost_sensitivity.csv")
    pred = pd.read_csv(output / "final_predictions.csv.gz", parse_dates=["timestamp"])
    selected = json.loads((output / "selected.json").read_text())
    charts = []
    captions = []
    for task in LABEL:
        d = dev.loc[dev.task == task].copy()
        d["configuration"] = d.bundle + " / " + d.horizon.astype(str) + "h"
        pivot = d.pivot_table(index="model", columns="configuration", values="balanced_accuracy", aggfunc="mean")
        charts.append(heatmap(pivot, f"{LABEL[task]} — geliştirme balanced accuracy (%)", figures / f"development_{task}.png"))
        captions.append("7 parite × 2 ileri dönem ortalaması. COT sütunları 14 günlük varsayımsal gecikme deneyidir; nihai model seçimine alınmadı.")
    ablation = dev.groupby(["task", "horizon", "bundle"]).balanced_accuracy.mean().unstack("bundle")
    ablation = ablation.subtract(ablation.price, axis=0)
    ablation.index = [f"{LABEL[t]} / {h}h" for t, h in ablation.index]
    ablation.to_csv(output / "feature_group_lift.csv")
    charts.append(heatmap(ablation, "Özellik grubu katkısı — fiyat modeline göre balanced accuracy farkı (puan)", figures / "feature_group_lift.png", center=0))
    captions.append("Aynı model aileleri ve dönemler eşit ağırlıklı. Artı değer geliştirmede iyileşme, eksi değer kötüleşme demektir; gelecekteki kazanç garantisi değildir.")
    f = final.pivot(index="symbol", columns="task", values="balanced_accuracy").rename(columns=LABEL)
    charts.append(heatmap(f, "Dondurulan modeller — 2023+ balanced accuracy (%)", figures / "final_balanced_accuracy.png"))
    captions.append("Parite bazında yeniden model seçimi yapılmadı. Her hedef için geliştirmede seçilen tek konfigürasyon tüm paritelere uygulandı. Örtüşmeyen örnekler.")
    for task in LABEL:
        d = final[final.task == task].set_index("symbol")
        frame = d[["accuracy", "majority_accuracy", "persistence_accuracy", "balanced_accuracy"]]
        frame.columns = ["Model accuracy", "Eğitim çoğunluğu", "Süreklilik kuralı", "Model balanced"]
        charts.append(heatmap(frame, f"{LABEL[task]} — model ve basit referanslar (%)", figures / f"baseline_{task}.png"))
        captions.append("Yön hedefinin süreklilik referansı geçmiş momentumdur; oynaklık hedefinin referansı mevcut oynaklık durumudur. Accuracy farklı hedefler arasında işlem başarısı olarak karşılaştırılmaz.")
    conf = final.copy()
    conf["label"] = conf.symbol + " / " + conf.task.map(LABEL)
    conf = conf.set_index("label")[["confident_accuracy", "confident_balanced_accuracy", "confident_baseline_accuracy", "confident_majority_accuracy", "confident_coverage"]]
    conf.columns = ["Seçilmiş accuracy", "Dengeli accuracy", "Süreklilik referansı", "Çoğunluk referansı", "Kapsam"]
    charts.append(heatmap(conf, "Geliştirmede dondurulan güven eşiği — accuracy ve kapsam (%)", figures / "confidence_accuracy.png", minimum=0, maximum=1))
    captions.append("Güven eşiği her geliştirme paritesi/döneminde en az %20 kapsam ve en az 30 örnek şartıyla seçildi. Bu sayı tüm örneklerdeki accuracy'nin yerine geçmez.")
    pred = pred[pred.nonoverlap].copy()
    pred["year"] = pred.timestamp.dt.year
    pred["correct"] = (pred.probability >= .5) == pred.truth
    year_rows = []
    from sklearn.metrics import balanced_accuracy_score
    for (task, symbol, year), group in pred.groupby(["task", "symbol", "year"]):
        if group.truth.nunique() == 2:
            year_rows.append({"task": task, "symbol": symbol, "year": year,
                "balanced_accuracy": balanced_accuracy_score(group.truth, group.probability >= .5)})
    yearly = pd.DataFrame(year_rows)
    yearly.to_csv(output / "yearly_metrics.csv", index=False)
    frame = yearly.groupby(["task", "year"]).balanced_accuracy.mean().unstack("year").rename(index=LABEL)
    charts.append(heatmap(frame, "Yıllara göre balanced accuracy — pariteler eşit ağırlıklı (%)", figures / "yearly_stability.png"))
    captions.append("2026 eksik yıldır. Pariteler birbirleriyle ilişkilidir; bunlar bağımsız tekrar deneyleri değildir.")
    cost = trades[trades.cost_pips == 1.5]
    pivot = cost.pivot(index="symbol", columns="threshold", values="mean_net_bps")
    charts.append(heatmap(pivot, "Yön modeli — işlem başına net baz puan, 1,5 pip maliyet", figures / "cost_heatmap.png", center=0, percent=False))
    captions.append("Eşikler final sonrası duyarlılık görselidir; buradan en iyi eşik seçilip doğrulanmış sonuç ilan edilmez. Boş hücre: işlem yok. Swap ve değişken spread dahil değil.")
    coverage = cost.pivot(index="symbol", columns="threshold", values="coverage")
    charts.append(heatmap(coverage, "Yön modeli — tahmin eşiğine göre işlem kapsamı (%)", figures / "coverage_heatmap.png", center=.5, minimum=0, maximum=1))
    captions.append("Yüksek seçilmiş accuracy az sayıda işlemden kaynaklanabilir; kapsam ve işlem sayısı birlikte değerlendirilmelidir.")
    bars = load_hourly(ROOT / "data/historical_1h/EURUSD.csv")
    x, groups = build_features(bars, "EURUSD")
    corr_cols = ["return_4", "return_24", "ema_20_atr", "rsi", "atr_pct", "rv_4", "rv_24", "rv_480",
                 "rv_ratio_4", "rv_ratio_24", "channel_24", "fomc_sentiment", "cot_base_net_oi", "cot_quote_net_oi"]
    correlation = x.loc[x.index < "2023-01-01", corr_cols].corr(method="spearman")
    charts.append(heatmap(correlation, "EURUSD geliştirme — özellik Spearman korelasyonu", figures / "feature_correlation.png", center=0, percent=False, minimum=-1, maximum=1))
    captions.append("Korelasyon nedensellik veya tahmin başarısı değildir. Yalnız geliştirme verisi kullanıldı; benzer bilgi taşıyan özellikleri gösterir.")
    importance = pd.read_csv(output / "feature_importance.csv")
    fig = go.Figure()
    for task in LABEL:
        part = importance[importance.task == task].nlargest(12, "importance")
        fig.add_trace(go.Bar(name=LABEL[task], x=part.feature, y=part.importance * 100))
    fig.update_layout(title="EURUSD — özellik karıştırıldığında balanced accuracy kaybı (puan)", template="plotly_dark", height=550)
    charts.append(fig)
    captions.append("Final verisinde yalnız açıklama amaçlı hesaplandı; özellik veya model seçimini değiştirmedi. Korelasyonlu özellikler tekil önemi paylaşabilir.")
    summaries = []
    for task in LABEL:
        d = final[final.task == task]
        summaries.append({"task": task, "pairs": len(d), "accuracy": float(d.accuracy.mean()),
            "balanced_accuracy": float(d.balanced_accuracy.mean()),
            "persistence_accuracy": float(d.persistence_accuracy.mean()),
            "majority_accuracy": float(d.majority_accuracy.mean()),
            "auc": float(d.auc.mean()), "n_nonoverlap": int(d.n.sum()),
            "positive_lift_ci_pairs": int((d.accuracy_lift_ci_low > 0).sum()),
            "confident_accuracy": float(d.confident_accuracy.mean()),
            "confident_balanced_accuracy": float(d.confident_balanced_accuracy.mean()),
            "confident_majority_accuracy": float(d.confident_majority_accuracy.mean()),
            "confident_baseline_accuracy": float(d.confident_baseline_accuracy.mean()),
            "confident_coverage": float(d.confident_coverage.mean()),
            "selected": selected[task]})
    (output / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    tiles = ""
    for s in summaries:
        tiles += f'<div class="tile"><h2>{LABEL[s["task"]]}</h2><strong>%{s["accuracy"]*100:.2f}</strong><p>Accuracy · balanced %{s["balanced_accuracy"]*100:.2f}</p><p>Süreklilik referansı %{s["persistence_accuracy"]*100:.2f}<br>{s["pairs"]} parite · {s["n_nonoverlap"]:,} örtüşmeyen örnek</p><p>Güven filtresi: %{s["confident_accuracy"]*100:.2f} accuracy / %{s["confident_coverage"]*100:.1f} kapsam</p></div>'
    sections = ""
    for i, (chart, caption) in enumerate(zip(charts, captions)):
        sections += '<section>' + chart.to_html(full_html=False, include_plotlyjs=True if i == 0 else False) + '<p>' + html.escape(caption) + '</p></section>'
    report = f'''<!doctype html><html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Forex ML Model Karşılaştırması</title>
<style>body{{background:#0b1220;color:#e4edf7;font:16px system-ui;margin:0 auto;max-width:1450px;padding:36px}}h1{{font-size:36px}}p{{line-height:1.7;color:#b8c6da}}.tiles{{display:flex;gap:20px;flex-wrap:wrap}}.tile,section{{background:#111c2d;border:1px solid #253750;border-radius:16px;padding:24px;margin:20px 0}}.tile{{flex:1;min-width:280px}}strong{{font-size:42px;color:#69d7c2}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:8px;border-bottom:1px solid #30425a}}.scroll{{overflow:auto}}code{{color:#8be0cc}}a{{color:#80c9ff}}</style></head><body>
<p>YEREL VERİ ARAŞTIRMASI · 2008–2026 · 1H</p><h1>Hangi ML yöntemi neyi öğrenebiliyor?</h1>
<p>Beş model ailesi, dört özellik grubu, üç tahmin ufku, iki hedef. Seçim yalnız 2016–2022 geliştirme dönemlerinde; 2023+ sonuçları seçim dondurulduktan sonra hesaplandı.</p>
<div class="tiles">{tiles}</div>
<p><b>İki hedef farklıdır:</b> yön accuracy'si fiyat yönünü, oynaklık accuracy'si gelecekteki hareketliliğin son 480 mumluk seviyeyi aşıp aşmayacağını ölçer. Oynaklık başarısı kârlı işlem oranı değildir. Ortalamalar paritelere eşit ağırlık verir.</p>
<p>EURUSD, GBPUSD ve USDJPY'nin son dönemleri önceki araştırmada görülmüştü. Diğer pariteler bu turnuvada seçim sonrası kontrol edildi. Yeniden deneme, parametre değiştirme veya en iyi final paritesini seçme bu kontrolü geliştirme verisine dönüştürür. Kaynak pariteler birbirinden bağımsız değildir.</p>
<p>COT yalnız varsayımsal gecikme deneyidir; FRED gözlem tarihleri ve sonradan oluşan haber etki sütunları kullanılmadı. Model çıktıları araştırmadır; canlı işlem izni veya kalibre başarı olasılığı değildir.</p>
{sections}<section><h2>Dondurulan konfigürasyonlar</h2><pre>{html.escape(json.dumps(selected, indent=2, ensure_ascii=False))}</pre></section>
<section><h2>Nihai ölçümler</h2><p>Accuracy avantajı güven aralıkları süreklilik referansına göre blok bootstrap ile hesaplandı. Parite bazlı aralıklar çoklu karşılaştırma düzeltmesi içermez; tekil pozitif aralık doğrulanmış işlem avantajı ilan edilmez.</p><div class="scroll">{final.to_html(index=False, float_format=lambda v: f"{v:.4f}")}</div></section>
<section><h2>Yöntem kaynakları</h2><p><a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html">Zaman sıralı doğrulama ve gap</a> · <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html">Dengeli doğruluk</a></p><p>Tekrar üretim: <code>python forex_ml_tournament.py</code>. Tamamlanan deney yeniden son test yapmadan kayıtlı raporu üretir.</p></section></body></html>'''
    (output / "report.html").write_text(report, encoding="utf-8")
    print(f"Report: {output / 'report.html'}", flush=True)


if __name__ == "__main__":
    generate()
