# Yerel verilerle ML araştırması

Bu ilk deneyin devamında beş model ailesiyle 1.680 geliştirme koşusu ve 28 parite
kontrolü tamamlandı. Güncel sonuçlar ve heatmap'ler: [ML_TOURNAMENT.md](ML_TOURNAMENT.md).

14 Eylül 2026 incelemesi: 257 CSV, 61.975.026 satır, 3.228.924.504 bayt.
Farklı zaman dilimleri aynı piyasayı tekrar temsil eder; satır sayısı bağımsız örnek sayısı değildir.
29 parite; 5m, 15m, 30m, 1h, 4h, 1d, 1w ve eski historical klasörü mevcut.
EURUSD 1H: 2008-01-01–2026-09-11, 114.653 mum.

Haber kaynakları: 3.253 makro olay, 22.435 olay/parite etki satırı,
157 FOMC açıklama skoru, 7.320 COT kaydı ve 20 FRED serisi.
Etki satırları bağımsız haber sayısı değildir. FOMC skorları sözlük tabanlıdır;
arşivde tam haber metni veya beklenti/gerçekleşen ekonomik takvim bulunmuyor.

## Çalıştırma

```powershell
python audit_ml_data.py
python forex_ml.py --symbols EURUSD GBPUSD USDJPY
python -m unittest -v test_forex_ml.py test_forex_decision_core.py test_forex_modules.py
streamlit run forex_web_app_streamlit_v14_alert_decision.py
```

Ana uygulamada “Makine öğrenmesi — araştırma sonuçları” bölümü kayıtlı raporları gösterir.
Eğitim çevrimdışıdır; ağ erişimi veya broker bağlantısı gerektirmez.
`data/ml/inventory.json`: dosya/tarih/satır/eksik hücre envanteri.
`data/ml/<PARITE>_report.json`: ayarlar, dönemler, ölçümler ve sınırlamalar.
`data/ml/<PARITE>_oos.csv`: eğitim dışında üretilmiş olasılıklar ve gerçekleşen ileri getiriler.
Aynı pariteyi tekrar çalıştırmak bu araştırma çıktılarını günceller.

## Deney tasarımı

Teknik özellikli gradient boosting ile aynı özelliklere FOMC skoru ve açıklamanın
yaşını ekleyen model karşılaştırılır. Hiperparametreler ve %60 eşiği sabittir.
Üç genişleyen eğitim/test dönemi, ardından son %20 holdout kullanılır.
Eğitim sonundan 24 mum çıkarılır: eğitim etiketleri test dönemine ulaşamaz.
Modelin rastgele iç doğrulamasını önlemek için early stopping kapalıdır.

Karar mum kapanışında, giriş sonraki mum açılışında, çıkış 24. mum kapanışındadır.
Etiket tüm gözlemlenebilir sonuçları içerir; zaman aşımı veya belirsiz bariyer
örnekleri seçilerek elenmez. İşlem değerlendirmesinde her 24 mumda bir ortak giriş
takvimi vardır; pozisyonlar örtüşmez. Varsayılan toplam işlem maliyeti 1,5 piptir.
Her mum için AUC/Brier, işlem takvimi üzerinde maliyet sonrası baz puan raporlanır.
Toplam baz puan bir portföy getirisi, kaldıraçlı performans veya risk düzeltilmiş R değildir.
Always-long ve eğitim sınıf oranı referansları raporda bulunur.

## İlk sonuçlar

Son %20 test dilimi, teknik + FOMC modeli:

| Parite | AUC | İşlem | İşlem başına net baz puan |
|---|---:|---:|---:|
| EURUSD | 0,5021 | 124 | -3,01 |
| GBPUSD | 0,5095 | 60 | +7,65 |
| USDJPY | 0,4983 | 97 | +6,12 |

Bu sonuçlar tutarlı ML üstünlüğünü doğrulamıyor. Üç paritede de holdout Brier,
eğitim sınıf oranı referansından kötü. Geliştirme dönemleri de karışık sonuçlar
veriyor. Olasılıklar kalibre edilmiş işlem başarı olasılıkları değildir.
Araştırma paneli mevcut TREND/RANGE kararlarını ve işlem iznini değiştirmez.
Mevcut motorların girişlerinde ek katkı henüz ölçülmedi.

## Veri sorunları ve kullanım sınırları

- NZDCHF 1H sıralı değil; eğitim yükleyicisi bunu reddeder. Kaynak değiştirilmedi.
- EURUSD 1H içinde 13 mumun sınırı kapanıştan bir fiyat adımı sapıyor.
  Yükleyici yalnızca bir fiyat adımına kadar OHLC sınırını bellekte genişletir,
  sayısını raporlar ve daha büyük bozukluğu reddeder. Kaynak dosya korunur.
- Envanter tarih/eksik hücre kontrolüdür; tüm dosyalar için broker doğruluğunu,
  eksik işlem saatlerini veya çapraz zaman dilimi tutarlılığını kanıtlamaz.
- FOMC açıklamalarında saat yok: ertesi UTC gece yarısından itibaren kullanılabilir
  varsayılır; 90 gün sonra bayat özellikler boş bırakılır.
- FRED gözlem tarihleri yayın tarihi değildir; revize geçmiş değerler ilk açıklanan
  değerlerle aynı olmayabilir. Events ve events_with_impact bu sebeple modele alınmadı.
- COT tarihleri pozisyonun ait olduğu gündür; doğrulanmış yayın zamanları olmadan alınmadı.
- Spread değişimi, swap, broker gerçekleşmesi ve tatil kaynaklı saat boşlukları
  gerçekçi işlem simülasyonu için ayrıca ele alınmalı. 24 mum her zaman 24 saat değildir.
- Eski `ml_poc_eurusd_1h.py` komutu güvenli olmayan deneyi çalıştırmak yerine yeni
  komuta yönlendirir; eski fonksiyonlar inceleme amacıyla korunur.

## Sonraki araştırma basamağı

Gerçek yayın zamanını (`published_at`, UTC), veri elde edilme zamanını, kaynak ve
ilk yayın değerini saklayan haber arşivi oluşturulmalı. COT yayın takvimi ve
FRED/ALFRED vintage değerleri bu aşamada eşleştirilmeli. Ardından mevcut TREND/RANGE
girişlerini etiketleyen meta-model, aynı girişlerde filtresiz motorla karşılaştırılmalı.
Karar eşiği yalnız geliştirme verisinde seçilmeli; bugünkü holdout görüldüğü için
sonraki geliştirmede yeni, dokunulmamış ileri dönem ayrılmalı. Model kaydı ve canlı
destek skoru, bu karşılaştırma ve olasılık kalibrasyonundan sonra eklenmeli.
