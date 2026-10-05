# Özet sekmesindeki dört görüş aynı yönde olunca — sonuç

5 Ekim 2026. Sonuç: geçmedi. Radar kartı, Fırsatlar, Pariteler ve ML yön modeli aynı yönü gösterdiğinde o yöne girmek, maliyetten önce yazı-tura, maliyetten sonra 4 saatte ortalama −2,2 pip. Bu yüzden Özet sekmesine LONG/SHORT diyen bir kutu eklenmedi. Onun yerine görüşleri ve bu ölçümü gösteren bir "Genel Yorum" paneli eklendi.

## Soru

Kullanıcı Özet sekmesinde AUDNZD için "LONG İÇİN İZLE 77/100", Fırsatlar'da "Güçlü Alım" ve ML'de "SHORT %59" gördü. Ne yapacağını bilemediği için ekranı okuyup LONG/SHORT diyen tek bir alan istedi. Bu dört görüşün her biri daha önce tek başına test edilmiş ve geçmemişti. Soru şuydu: dördü birleşince, yani hepsi aynı yönü gösterince, o yöne girmek maliyetten sonra kazandırıyor mu?

## Kurgu

- **Görüşler:** Uygulamanın kendi kodu (`analyse_symbol`, `global_bias`, `build_intraday_opportunity`) geçmişte her saat yeniden çalıştırıldı. Sadece veri kaynağı değişti: her anda, uygulamanın o an indireceği mumlar verildi (feed_replay.py).
  - **Radar kartı:** İZLE veya HAZIR gösteriyor (puan ≥ 42).
  - **Fırsatlar:** 15M veya 5M girişi Güçlü Alım/Satış.
  - **Pariteler:** 4H, 1H, 15M ve 5M'nin dördü de aynı yönde.
  - **ML:** 4 saatlik yön modelinin olasılığı %50'nin üstünde veya altında.
- **Dönem:** ML modelleri 2023 öncesiyle eğitildi. Bu yüzden test 2023 ve sonrasını kapsıyor; ML'nin cevabı önceden görmediği tek dönem bu.
- **Sonuç ölçüsü:** 1 ve 4 saat sonraki hareket (pip). Ölçülen spread + 0,7 pip komisyon düşüldü.
- **Geçme şartı:** Ortalama net hareket hem 1 hem 4 saatte artıda ve %97,5 aralığı sıfırın üstünde olmalı. Ayrıca 4 saatlik sonuç her alt dönemde artıda olmalı.
- **Kontrol:** Her parite-yılda rastgele 20 an sıfırdan yeniden hesaplandı. Görüşler birebir, ML olasılığı canlı uygulamanın yoluyla (`build_research_prediction`) 1e-9 içinde tuttu.
- Kurallar sonuç hesaplanmadan protocol.json dosyasına yazıldı.

## Protokolden sapma: erken durduruldu

Tam çalıştırma 28 parite × 2023–2026 için yaklaşık 6 saat sürecekti. İlk dalga bittiğinde (14 parite, 2023 yılı, 85.839 saatlik an) bir ara bakış yapıldı. Kullanıcı kalanını beklememeyi seçti ve çalıştırma durduruldu.

- Sonuçlar bu 14 paritenin 2023 yılına dayanıyor.
- Alt dönem şartı (2024 ve 2025–2026) değerlendirilemedi.
- Bu, sonucu değiştirmiyor. Ana şart zaten tek başına geçmedi: net aralıklar tamamen sıfırın altında.

## Sonuçlar

**Hipotez** (dört görüş aynı yönde, 12.234 an, tüm anların %14'ü):

| | 1 saat | 4 saat |
|---|---|---|
| Maliyetten önce hareket | −0,03 pip | +0,24 pip |
| Net (maliyetten sonra) | −2,42 [−2,76, −2,08] | −2,15 [−3,26, −1,05] |
| Kazanma oranı | %48,9 | %50,3 |

Geçmedi.

**Görüşler tek tek ve birlikte** (4 saat; pip):

| Durum | An | Hareket | Net | Kazanma |
|---|---:|---:|---:|---:|
| Radar kartı | 59.102 | −0,20 | −2,74 | %49,0 |
| Fırsatlar | 36.378 | −0,45 | −3,02 | %48,8 |
| Pariteler | 31.726 | −0,44 | −2,98 | %48,2 |
| ML yön | 85.839 | +0,59 | −1,94 | %52,0 |
| Üç teknik görüş aynı yönde | 28.419 | −0,44 | −2,98 | %48,2 |
| … ML de aynı yönde (dördü birden) | 12.234 | +0,24 | −2,15 | %50,3 |
| … ML ters yönde (teknik yöne girilirse) | 16.185 | −0,96 | −3,61 | %46,6 |
| Dördü birden, yeni başladığı an | 5.837 | +0,17 | −2,25 | %50,2 |

Ortalama maliyet 2,53 pip.

## Yorum

- **Hepsinin aynı yönü göstermesi avantaj değil.** Maliyetten önce sonuç yazı-tura, maliyetten sonra her işlem ortalama 2 pip zararda.
- **Teknik görüşler maliyetten önce bile hafif eksi.** Radar, Fırsatlar ve Pariteler'in yönünde girmek 4 saatte 0,2–0,45 pip, 24 saatte 2–2,6 pip kaybettiriyor. Bu, 15M/5M sinyallerinin önceki testleriyle örtüşüyor.
- **Yön bilgisi taşıyan tek görüş ML.** %52 kazanma ve 4 saatte +0,6 pip, daha önce ölçülen isabetle tutarlı. Ama maliyetin yanında çok küçük.
- **Teknik ve ML çeliştiğinde** teknik yöne girmek en kötü durum: %46,6 kazanma, −3,6 pip. Kullanıcının AUDNZD ekranı (teknik LONG, ML SHORT) tam bu durumdu; LONG'a girmemek doğruydu. Ama ML'nin yönüne girmek de maliyeti karşılamazdı.
- **Protokolde olmayan bir gözlem:** Teknik görüşlerin 24 saatte ters yöne dönmesi (−2,6 pip) dikkat çekici. Ancak sonuçlara bakarak bulundu; kural yapılmadı. Sınanacaksa yeni ve önceden yazılmış bir testle sınanmalı.

## Uygulamaya

Protokoldeki "geçmezse" adımı uygulandı. Özet sekmesine, radar kartının altına "🧭 Genel Yorum" paneli eklendi (forex_overall.py):

- Dört görüşün ne dediğini yan yana gösterir.
- Durumu adlandırır: dördü aynı yön / teknik ile ML çelişiyor / kısmi / yön yok.
- O durumun bu testteki sonucunu yazar.
- Karar satırı her zaman "Yeni işlem açma"; nedeni de yanında yazılı.
- Yine de girilirse dikkat edilecekleri sıralar: 4 saat içinde paritenin para birimlerinde yüksek etkili haber, maliyetin planlanan kayba oranı (%15 üstü) ve 72 saatlik oynaklık modeli "yüksek" diyorsa.

## Dosyalar

- protocol.json — önceden yazılan kurallar (erken durdurma notu dahil)
- results.json — bütün sayılar (14 parite, 2023)
- ../../research_consensus.py — `build` ve `evaluate` aşamaları
- ../../test_research_consensus.py — görüş kurallarının testleri
- ../../forex_overall.py, ../../test_forex_overall.py — panel ve kurallarının araştırmayla aynı olduğunu sınayan testler
