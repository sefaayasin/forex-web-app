# Forex Analyzer Pro

Geniş model karşılaştırması, heatmap'ler ve otomatik tester:
[ML_TOURNAMENT.md](ML_TOURNAMENT.md). Çalıştırma: `python forex_ml_tournament.py`.

Yerel veri envanteri ve zaman sıralı makine öğrenmesi araştırması için
[ML_RESEARCH.md](ML_RESEARCH.md) dosyasına bakın. Eğitim:
`python forex_ml.py --symbols EURUSD GBPUSD USDJPY`.
Sonuçlar uygulamadaki “Makine öğrenmesi — araştırma sonuçları” bölümünde görünür.

Streamlit tabanlı çoklu zaman dilimi forex karar destek, risk planlama ve backtest uygulaması.

> Bu proje yatırım tavsiyesi değildir. Gerçek işlemlerden önce demo hesapta ve broker verisiyle doğrulanmalıdır.

## Çalıştırma

```bash
pip install -r requirements.txt
streamlit run forex_web_app_streamlit_v14_alert_decision.py
```

## Modüler yapı

- `forex_web_app_streamlit_v14_alert_decision.py`: Streamlit giriş noktası ve ekran akışı.
- `forex_config.py`: pariteler, zaman dilimleri, seanslar ve pip kuralları.
- `forex_indicators.py`: RSI, MACD, Bollinger, ATR, Ichimoku ve EMA hesapları.
- `forex_analysis.py`: piyasa yapısı, corrective/response fazı ve yön skoru.
- `forex_decision_core.py`: dış bağımlılığı olmayan sinyal, pozisyon seviyesi ve istatistik kuralları.
- `forex_edge.py`: bootstrap/circular-shift edge doğrulama orkestrasyonu ve raporu.
- `forex_storage.py`: SQLite işlem günlüğü, alarm tekilleştirme ve webhook gönderimi.

Modül testleri:

```bash
python -m unittest -v test_forex_decision_core.py test_forex_modules.py
```

## Streamlit Community Cloud

- Repository: `sefaayasin/forex-web-app`
- Branch: `main`
- Main file path: `forex_web_app_streamlit_v14_alert_decision.py`

ML sonuçları için sol menüden `ML Laboratuvarı` seçin veya uygulama adresine
`?view=ml` ekleyin. Hazır raporlar ve heatmap'ler depoya dahildir; sunucuda
yeniden eğitim yapılmaz. Ham fiyat/haber arşivi ve büyük model/tahmin dosyaları
yerelde tutulur. Eski `forex_web_app_streamlit.py` giriş yolu da güncel uygulamayı açar.

Community Cloud ortamında MetaTrader 5 terminali çalışmaz. Uygulama veri kaynağını sessizce değiştirmez; Yahoo, yerel MT5 veya Broker CSV açıkça seçilir. SQLite günlükleri bulut yeniden başlatmalarında kalıcı olmayabilir.

## Rejim uyumlu çift motor

- `TREND`: 4H/1H yönü içinde 15M düzeltme + tepki devam modeli.
- `RANGE`: yalnız yatay rejimde bant ihlali sonrası Bollinger orta banda dönüş modeli.
- Motorların backtest kalitesi ayrı hesaplanır. Doğrulanmayan motor diğer rejimden sinyal ödünç alamaz.
- `Trend + Yatay Motoru ve Edge'i Test Et` düğmesi iki motoru aynı maliyet ve risk koşullarında karşılaştırır.

## Edge doğrulama

- Yüksek indikatör skoru kazanma olasılığı olarak yorumlanmaz.
- Stationary bootstrap, işlem başına ortalama `R` sonucunun sıfırın gerçekten üzerinde olup olmadığını sınar ve %95 güven aralığını raporlar.
- Circular-shift testi, gerçek giriş zamanlarını aynı LONG/SHORT dizisinin rastgele kaydırılmış zamanlarıyla karşılaştırır.
- Ana edge kapısı; en az 60 işlem, pozitif ortalama R, sıfırın üzerinde %95 alt güven sınırı, pozitif son-%30 OOS ortalama R ve düzeltilmiş bootstrap `p ≤ 0.05` şartlarını birlikte arar.
- Circular-shift sonucu ayrı bir `Zamanlama Teyidi` olarak raporlanır; aynı örnekten üretilen ikinci bir zorunlu p-kapısı değildir.
- Bonferroni çarpanı çift motor laboratuvarında kullanıcı tarafından değiştirilemez: seçili parite için önceden tanımlı `TREND + RANGE = 2` hipotezdir.
- Kısmi fakat pozitif kanıt `ADAY / DEMO` olarak gösterilir; bu seviye gerçek işlem izni vermez.
- Canlı işlem izni için aktif motorun backtest kalitesi `Orta/İyi` ve edge sonucu `DOĞRULANDI` olmalıdır.

## 6–12 aylık broker testi

Yerel Windows + MT5 bağlantısında veri kaynağını `MetaTrader 5`, araştırma periyodunu `180d`–`365d` seçin. Alternatif olarak MT5 geçmişini CSV dışa aktararak Streamlit Cloud'da `Broker CSV` kaynağını kullanabilirsiniz.

CSV için gereken alanlar:

```text
date + time (veya tek time), open, high, low, close
```

İsteğe bağlı `tickvol`/`volume` ve `spread` alanları desteklenir. CSV'nin gerçek mum zamanını (`1m`, `5m`, `15m`) doğru seçin. Uygulama daha yüksek zaman dilimlerini kendisi üretir; daha düşük zaman dilimini tahmin etmez.
