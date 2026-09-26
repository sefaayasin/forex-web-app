# FOMC günü dolar etkisi — yayın sonrası test

26 Eylül 2026. Sonuç: Mueller, Tahbaz-Salehi & Vedolin'in (Journal of Finance, 2017) bulgusu kendi döneminde bu veride de çok güçlü çıkıyor. Ama makalenin verisinin bittiği 2014'ten sonra, önceden yazılan anlamlılık ölçütünü geçmedi. Uygulamaya bir şey eklenmedi.

## Test edilen iddia

Doları diğer büyük para birimlerine karşı satan bir portföy, planlı FOMC açıklama günlerinde diğer günlerden belirgin biçimde fazla kazanıyor. Makalede (1994–2013) bu basit strateji açıklama günlerinde ortalama 10,8 baz puan kazanıyor; AUD'de fark 16 baz puan (t=2,5).

## Kurgu

- **Ana dönem:** 2014 – 11 Eylül 2026, yani makalenin hiç görmediği dönem. 96 planlı FOMC günü, 3.022 diğer gün.
- **Kontrol dönemi:** 2008–2013. Makalenin dönemiyle çakışıyor; amaç sadece verimizin ve yöntemimizin etkiyi yeniden üretip üretmediğini görmek.
- **Para birimleri:** EUR, GBP, AUD, NZD, JPY, CAD, CHF — makalenin G10 setinden, arşivde olmayan NOK ve SEK çıkarılarak. Eşit ağırlıklı ve dolar karşısında alınmış.
- **Günlük getiri:** New York saatiyle 16:00'dan ertesi iş günü 16:00'ya (makalenin yöntemi). Açıklama 14:00'te, yani bu pencerenin içinde.
- **FOMC günleri:** Fed'in resmi takviminden doğrulanan planlı toplantıların son günü. Olağanüstü açıklama günleri (2008 acil indirimleri, 2020 Mart kararları ve benzerleri) iki gruptan da çıkarıldı. Dosyada eksik olan 2008-12-16 eklendi.
- **Kapsam dışı:** Faiz farkı (carry). Her iki gruba da aynı küçük günlük getiriyi eklediği için aradaki farkı değiştirmez.
- Kurallar ve ölçütler sonuç hesaplanmadan önce protocol.json dosyasına yazıldı.

## Sonuçlar

| Dönem | FOMC günü ort. | Diğer günler ort. | Fark | Welch t | Tek yönlü permütasyon p |
|---|---:|---:|---:|---:|---:|
| 2008–2013 | +24,1 bp | −1,0 bp | **+25,1 bp** | 1,93 | **0,001** |
| **2014–2026 (ana)** | +3,9 bp | −0,9 bp | **+4,8 bp** | 0,80 | **0,13** |
| 2014–2019 | +4,7 bp | −1,2 bp | +5,9 bp | 0,64 | 0,15 |
| 2020–2026 | +3,2 bp | −0,5 bp | +3,7 bp | 0,48 | 0,28 |

Ana dönemde maliyet sonrası FOMC günü ortalaması: 1,5 pip ile +2,4 bp, 3 pip ile +0,8 bp.

| Ölçüt | Sonuç |
|---|---|
| Ana dönemde fark anlamlı (p ≤ 0,05) | **Geçmedi** (p = 0,13) |
| Ana dönemde 1,5 pip maliyet sonrası artı | Geçti (+2,4 bp) |
| İki alt dönemde de fark artı | Geçti (+5,9 ve +3,7 bp) |

Yıllara göre FOMC günü ortalaması çok dalgalı: 2014'te −41 bp, 2020'de +33 bp, 2023'te +32 bp, 2025'te −26 bp, 2026'da (5 toplantı) −39 bp.

## Değerlendirme

- **Etki kendi döneminde gerçek, yayından sonra zayıflamış.** Veri ve yöntemimiz 2008–2013'te 25 baz puanlık, çok anlamlı bir fark üretiyor. 2014 sonrasında fark yaklaşık beşte bire iniyor ve gürültüden ayrılamıyor. Bu, yayımlanan birçok piyasa anomalisinin yayından sonra küçüldüğü bulgusuyla uyumlu.
- **Ekonomik büyüklük küçük.** Yılda 8 toplantıda işlem başına yaklaşık +2,4 bp (1,5 pip maliyetle), yani yılda kabaca 0,2 puan. Tek bir FOMC gününde basketin hareketi ±40 bp olabildiği için, perakende bir trader için anlamlı bir strateji değil.
- **Açıklama öncesi / sonrası ayrımı** (bilgi amaçlı; ölçütlere dahil değil):
  - 16:00'dan 14:00'e (açıklama öncesi): FOMC günlerinde +7,5 bp fark, t=2,3, p=0,03.
  - 14:00'ten 16:00'ya (açıklama sonrası): −2,8 bp.

  Açıklamadan önce dolar zayıflıyor, sonra kısmen geri alıyor. Bu, makalenin "sıkılaştırma dönemlerinde açıklama sonrası getiri düşük" bulgusuyla uyumlu; 2014 sonrası iki faiz artırım döngüsü içeriyor. Ancak bu ayrım birden çok testten biri ve sonuçlar görüldükten sonra öne çıkarılıyor. Strateji sayılmadan önce 2026 sonrası yeni FOMC günleriyle önceden yazılmış ayrı bir testten geçmeli.

## Sınırlar

- NOK ve SEK arşivde yok. Faiz farkı ve gecelik swap dahil değil.
- Ana dönemdeki 96 gün, küçük bir etkiyi yakalamak için az. p = 0,13, "etki yok" değil, "kanıtlanamadı" demek.
- Maliyet sabit varsayıldı. FOMC günü akşamı spread genişler ve pozisyon açıklamadan önceki akşam açılıyor; gerçek maliyet muhtemelen daha yüksek.

## Tekrar çalıştırma

`python -X utf8 research_fomc_day.py`

data/historical_1h_from_15m gerekir (`python rebuild_hourly_from_15m.py`). results.json bütün testleri, kullanılan FOMC tarihlerini ve para birimi bazında sonuçları içerir; fomc_basket_by_year.csv yıllık ortalamaları gösterir.
