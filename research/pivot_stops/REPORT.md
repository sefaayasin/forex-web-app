# Günlük pivot çizgileri — fiyat orada durur mu? — sonuç

20 Ekim 2026. Sonuç: hipotez geçmedi. Fiyat klasik günlük pivot çizgilerine (P, R1, R2, S1, S2) gelince, rastgele kaydırılmış sahte çizgilere geldiğindekinden daha sık geri dönmüyor. Çizgiler anasayfadaki mum grafiğine yalnızca bilgi olarak eklendi ve altına bu sonuç yazıldı.

## Soru

Kullanıcı kârdaki bir işlemde (GBPUSD +25 $, hedef 100 $) fiyatın devam mı edeceğini, yoksa bir çizgide durup geri mi döneceğini bilmek istedi. Soru: fiyat bir pivot çizgisine gelince, pivot olmayan bir çizgiye geldiğindekinden daha sık geri dönüyor mu?

## Kurgu

- **Veri:** 28 parite (EURZAR hariç), 2 Ocak 2008 – 11 Eylül 2026.
  - Seviyeler Dukascopy'nin kendi günlük mumlarından (UTC gün sınırı) hesaplandı.
  - Dokunuşlar ve sonuçlar 15 dakikalık mumlardan.
- **Seviyeler:** `forex_pivots.pivot_levels`, uygulamanın kullandığı fonksiyonun aynısı. Önceki hafta içi günün mumundan:
  - P = (H + L + C) / 3
  - R1 = 2P − L, S1 = 2P − H
  - R2 = P + (H − L), S2 = P − (H − L)
  - Pazartesi Cuma'nın mumunu kullanır.
- **Sahte çizgiler:** Her gerçek seviye, önceki günün aralığının ±0,25 ve ±0,5 katı kaydırıldı. Sahte çizgiler aynı kurallarla işlendi.
- **Dokunuş:** Önceki 15M kapanış çizginin altındayken mumun en yükseği çizgiye ulaştı (yukarı dokunuş); ya da tersi (aşağı dokunuş). Her çizgi ve her yön için günde yalnızca ilk dokunuş sayıldı.
- **Sonuç:** Dokunuştan sonra 4 saat içinde hangisi önce oldu?
  - Fiyat çizginin 1 ATR ötesine geçti → **geçti**.
  - Fiyat çizginin 1 ATR gerisine döndü → **döndü**.
  - Hiçbiri olmadıysa çözülmedi sayıldı (dokunuşların %3,2'si).
- **İstatistik:** Takvim günlerini bloklar halinde yeniden örnekleyen bootstrap (2.000 tekrar), %95 aralık.
- **Geçme şartı:** Gerçek − sahte dönme oranı farkının aralığı sıfırın üstünde olmalı. Fark üç alt dönemin (2008–2015, 2016–2023, 2024–2026) her birinde de pozitif olmalı.
- **Kontrol:** Her paritede 200 rastgele mumda, araştırmanın hızlı seviye hesabı modülün `levels_for_day` fonksiyonuyla birebir tuttu.
- Bütün kurallar sonuç hesaplanmadan önce protocol.json dosyasına yazıldı. Hiçbir parametre sonuçlara bakılarak seçilmedi.

## Sonuçlar

| | Dokunuş | Döndü | Geçti |
|---|---:|---:|---:|
| Gerçek pivot çizgileri | 442.551 | %48,16 | %51,84 |
| Sahte çizgiler | 1.819.580 | %48,07 | %51,93 |

**Fark:** +0,08 puan, %95 aralık [−0,05, +0,22]. **Geçmedi.**

| Dönem | Gerçek | Sahte | Fark |
|---|---:|---:|---:|
| 2008–2015 | %48,99 | %48,88 | +0,12 |
| 2016–2023 | %47,78 | %47,76 | +0,02 |
| 2024–2026 | %46,97 | %46,76 | +0,21 |

**Seviyelere göre** (bilgi; çoklu karşılaştırma düzeltmesi yok):

| Seviye, dokunuş yönü | Dokunuş | Gerçek döndü | Sahte döndü |
|---|---:|---:|---:|
| R1, fiyat yukarı çıkarken | 52.984 | %47,95 | %48,06 |
| R2, fiyat yukarı çıkarken | 20.791 | %47,63 | %47,75 |
| P, fiyat yukarı çıkarken | 90.337 | %48,20 | %47,82 |
| P, fiyat aşağı inerken | 90.210 | %49,11 | %48,23 |
| S1, fiyat aşağı inerken | 51.947 | %47,35 | %48,71 |
| S2, fiyat aşağı inerken | 21.136 | %47,71 | %47,31 |

28 paritenin 16'sında gerçek çizgilerde dönme oranı sahtelerden biraz yüksek, 12'sinde düşük. GBPUSD: gerçek %47,4, sahte %47,1.

## Yorum

- **Pivot çizgileri fiyatı durdurmuyor.** Gerçek ve sahte çizgilerde fiyat aynı sıklıkla dönüyor (%48). Fark yarım puan bile değil.
- **Bir çizgiye gelen fiyat daha çok devam ediyor.** Gerçek ya da sahte, her çizgide fiyatın %52'si devam etti, %48'i döndü. Bu çizgiden değil, 15M fiyatının kısa vadeli devam eğiliminden geliyor.
- **Kullanıcının sorusu için:** Yukarı giden bir işlemde fiyat R1'e geldiğinde 100 seferin yaklaşık 48'inde geri döndü, 52'sinde devam etti. Rastgele bir çizgide de aynısı oldu. Çizgiye bakıp "burada duracak" demenin ölçülmüş bir dayanağı yok.
- **P'de küçük bir fark var ama güvenilir değil.** Fiyat P'ye yukarıdan inerken dönme oranı sahteden 0,9 puan yüksek. Bu, sonuçlara bakıldıktan sonra on gruptan seçilmiş bir fark ve test edilen hipotez değil.

## Sınırlar

- Uygulama Yahoo'nun günlük mumlarını kullanıyor; Yahoo'nun gün sınırı UTC'den bir saat sapabilir (Londra saati). Araştırma Dukascopy UTC günleri kullandı.
- Brokerların çoğu günü New York 17:00'de kapatır; onların pivotları biraz farklı çıkar. Bu test edilmedi.
- Adım tek bir değerle (1 ATR, 4 saat) denendi. Haftalık/aylık pivotlar ve Camarilla/Fibonacci pivotları test edilmedi.

## Uygulamaya

Anasayfadaki mum grafiğine günlük pivot çizgileri ve fiyatın her birine kaç pip uzakta olduğu eklendi. Altında tek satırlık not var: fiyat bu çizgilerde rastgele çizgilerden daha sık durmuyor.

## Dosyalar

- protocol.json — önceden yazılan kurallar
- results.json — bütün sayılar (seviye, dönem ve parite dökümü dahil)
- ../../research_pivot_stops.py — `build` ve `evaluate` aşamaları
- ../../forex_pivots.py — uygulama ve araştırmanın ortak pivot hesabı
- ../../test_forex_pivots.py, ../../test_research_pivot_stops.py — birim testleri
