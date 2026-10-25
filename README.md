# Forex Analyzer Pro

Anasayfa → Pivot Trader → **İşlem takip grafiği**: 5M mumlarının üzerinde giriş,
kırmızı stop alanı ve yeşil hedef alanı gösterilir. Hazır planı **Bu planı takibe al**
ile sabitleyin veya **Kendi planını ekle** bölümüne seviyelerinizi girin. Takipteki
planın seviyeleri yeni sinyallerle değişmez; aynı veri kaynağındaki kayıtlar
parite taramadan çıkarılsa da saklanır. **Takibi bitir** yalnız görsel takibi
arşivler. Fiyat hareketi gösterge niteliğindedir, broker kâr/zararı veya TP/SL
gerçekleşmesi değildir. Yerel SQLite kayıtlarını **Takip listesini indir** ile
saklayabilirsiniz; bulut sunucusu yenilenirse yerel kayıtlar kaybolabilir.

Pivot Trader canlı alanı küçük net hedefle çalışır: koşullar uygunsa `LONG GİR` /
`SHORT GİR`, eksikse `BEKLE`. Referans giriş, net 3 pip hedefi, stop, lot ve
planlanan dolar kaybı birlikte gösterilir. Panel 30 saniyede bir yenilenir;
eski fiyat veya kaçan giriş bölgesi yeni giriş önerisini engeller. İşlem başına
zarar bütçesi panelden seçilir. GİR kararı zarar etmeme garantisi değildir.
[Küçük hedef denemesi](research/pivot_advisor_small_profit/REPORT.md): 12/12
geçmiş örnek, bağımsız doğrulama veya gelecekteki kazanma olasılığı değildir.

Pivot Trader artık açıklamalı işlem yorumcusu olarak çalışır. Özet ekranında
izlenecek pariteleri seçip `Yorumcuyu yenile` düğmesine basın: fiyat, zaman
dilimleri, ML ve takvim birlikte tekrar okunur. Her paritede yön, 8 koşulun
uyum puanı, giriş/stop/hedef izleme planı ve bekleme nedenleri gösterilir.
70/100 teknik uyum **%70 kazanma ihtimali değildir**. Güncel koşullar
tamamsa `LONG GİR` / `SHORT GİR`, eksikse `BEKLE` kararı üretilir.
Ana trend, tepki, giriş bölgesi, veri, haber ve maliyet kontrolleri zorunludur;
güncel ML güçlü biçimde ters yöndeyse giriş bekletilir. Stop ve hedef/risk
oranı güncel gösterge fiyatından hesaplanır. Birleşik kuralın bağımsız
başarı testi henüz bulunmuyor. Eski pivot kuralının
maliyet sonrası sonucu negatiftir; ML yön isabeti de yaklaşık %52'dir.
Mevcut araştırmalar, bu yeni birleşimin başarı kanıtı olarak kullanılmaz.
Yeni yorumcunun kendi kurallarıyla yapılan geçmiş test:
[Pivot Trader v2 raporu](research/pivot_advisor/REPORT.md).
1 Ekim 2024–11 Eylül 2026, yedi ana parite: 974.306 karar anında 19 giriş
kararı, tekrarlar/eksik yol ayrılınca 11 işlem. Maliyet sonrası 4 kâr, 6 stop,
hedefe ulaşan işlem yok; ortalama −7,52 pip/işlem. İki tekrarsız LONG'da
1 ve 4 saat sonraki yön doğru, fakat biri önce stop oluyor. Bu küçük örnek
%70 başarıyı veya kârlılığı doğrulamıyor.
[Tüm işlemler](research/pivot_advisor/all_trades.csv) giriş, stop, hedef ve sonucu içerir.
Küçük kârla çıkma senaryosu ayrıca test edildi:
[küçük net hedef raporu](research/pivot_advisor_small_profit/REPORT.md).
Aynı sinyallerde net 3 pip hedefi 12/12, net 5 pip hedefi 10/11 işlemde stop
öncesi görüldü. Bu çok küçük ve önceden incelenmiş örnek, zarar etmeme veya
gelecekte aynı başarı garantisi değildir; dolar sonuçları lot ve maliyete bağlıdır.
Tekrar çalıştırma: `python -X utf8 research_pivot_advisor.py` (yerel arşiv gerekir;
protokol kaynak/veri değişikliklerini kontrol eder).
Kontroller: `python -m unittest test_forex_pivot_advisor test_forex_pivot_plan test_forex_pivots test_forex_trade_plan test_forex_ml_live test_forex_costs`.

30/100 hedef yapısı için yeni aramalar (gece seansı, günlük, hafta sonu boşluğu,
COT, 2010–2026 haber sürprizi, 30 yeni indikatör 15M/1H/4H, altın; hiçbiri geçmedi):
[rapor](research/bracket_search/REPORT.md).

Arkadaşın MT5 projesindeki kurallar ve ekran fikirleri (USDJPY H4 geri çekilme,
altın M15, yön merdiveni, sert hareket uyarısı, haber geçmişi, parçalı çıkış)
görülmemiş yıllarda sınandı; hiçbiri geçmedi:
[rapor](research/arda_ideas/REPORT.md).

İnternetten araştırılan yedi yeni indikatörün ve fiyat dışı haber sürprizi
yönteminin sonuçları: [yeni araştırma raporu](research/new_indicators/REPORT.md).
ML ekranı kaynakları ve parite testlerini gösterir. Haber kaydı:
`python -X utf8 forex_news_recorder.py --watch 600` (broker emri göndermez).

1000 dolar hesap / 30 dolar net hedef / 100 dolar planlanan stop senaryosu:
[hedefe göre test ve teknik düzeltmeler](research/cash_target/REPORT.md).
Canlı/backtest skorları ortak hesap kullanır; uzun testler üst zaman diliminin
tam geçmişini ister. Negatif veya doğrulanmamış işlem planı giriş önermez.
Hedefe uygun otomatik öğrenme: `python -X utf8 forex_adaptive.py --net-target-ratio 0.3 --output research/adaptive_cash_entry --watch 600`.

15 dakikalık girişler için otomatik model/indikatör seçimi, öğrenilen ağırlıklar
ve üç aylık geçmiş testleri eklendi. Sonuçlar ve sınırlar:
[otomatik öğrenme raporu](research/adaptive_entry/REPORT.md).
Çalıştırma: `python -X utf8 forex_adaptive.py`; yerel arşivdeki değişiklikleri
izleyerek yeniden öğrenme: `python -X utf8 forex_adaptive.py --watch 600`.
ML Tahmini sekmesinde raporlar ve yerel güncelleme butonu bulunur.
Yeni öğrenme deneyi maliyet sonrası üstünlük göstermediği için mevcut
işlem kurallarını değiştirmez.

Günlük kullanımda sol menüden parite ve işlem riskini seçin. `Planı Kontrol Et`
normal backtest ile trend/yatay motor kanıtı testini birlikte çalıştırır; bu işlem
tek backtestten uzun sürebilir. Otomatik plan kontrolü yalnız normal backtesti
yeniler. Motor testi eksikse ana ekranda ayrıca belirtilir.

Ana ekran yönü, teknik giriş durumunu ve strateji kontrolünü ayrı gösterir.
`Sinyali ne engelliyor?` son hesaplamadaki filtre sonuçlarını açıklar; tarihsel
filtre katkısı ölçümü değildir. Teknik adaylar işlem alarmı sayılmaz.
Detaylı seçenekler `Ayarlar ve araştırma araçları` altında, veri kaynağı ise
sol menüdeki `Bağlantı ve grafik` bölümündedir. `Gelişmiş görünüm` teknik
görünümü açar. Sinyal eşikleri ve varsayılan teknik filtreler korunmuştur.

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

Parite başına ML tahmini ve Parite Alarm Ekranı, `🔥 Fırsat Akışı` sekmesinin
alt sekmelerindedir. Hazır raporlar ve heatmap'ler depoya dahildir; sunucuda
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
