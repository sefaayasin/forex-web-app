# Forex Analyzer Pro

Streamlit tabanlı çoklu zaman dilimi forex karar destek, risk planlama ve backtest uygulaması.

> Bu proje yatırım tavsiyesi değildir. Gerçek işlemlerden önce demo hesapta ve broker verisiyle doğrulanmalıdır.

## Çalıştırma

```bash
pip install -r requirements.txt
streamlit run forex_web_app_streamlit_v14_alert_decision.py
```

## Streamlit Community Cloud

- Repository: `sefaayasin/forex-web-app`
- Branch: `main`
- Main file path: `forex_web_app_streamlit_v14_alert_decision.py`

Community Cloud ortamında MetaTrader 5 terminali çalışmaz. Uygulama veri kaynağını sessizce değiştirmez; Yahoo, yerel MT5 veya Broker CSV açıkça seçilir. SQLite günlükleri bulut yeniden başlatmalarında kalıcı olmayabilir.

## Rejim uyumlu çift motor

- `TREND`: 4H/1H yönü içinde 15M düzeltme + tepki devam modeli.
- `RANGE`: yalnız yatay rejimde bant ihlali sonrası Bollinger orta banda dönüş modeli.
- Motorların backtest kalitesi ayrı hesaplanır. Doğrulanmayan motor diğer rejimden sinyal ödünç alamaz.
- `Trend + Yatay Motoru Test Et` düğmesi iki motoru aynı maliyet ve risk koşullarında karşılaştırır.

## 6–12 aylık broker testi

Yerel Windows + MT5 bağlantısında veri kaynağını `MetaTrader 5`, araştırma periyodunu `180d`–`365d` seçin. Alternatif olarak MT5 geçmişini CSV dışa aktararak Streamlit Cloud'da `Broker CSV` kaynağını kullanabilirsiniz.

CSV için gereken alanlar:

```text
date + time (veya tek time), open, high, low, close
```

İsteğe bağlı `tickvol`/`volume` ve `spread` alanları desteklenir. CSV'nin gerçek mum zamanını (`1m`, `5m`, `15m`) doğru seçin. Uygulama daha yüksek zaman dilimlerini kendisi üretir; daha düşük zaman dilimini tahmin etmez.
