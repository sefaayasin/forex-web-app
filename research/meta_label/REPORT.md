# 15M + 5M sinyalinin kendisini tahmin etmek (meta-etiketleme) — sonuç

26 Eylül 2026. Sonuç: model önceden yazılan beş ölçütten ikisini geçemedi ve uygulamaya eklenmedi. Seçtiği sinyaller bütün sinyallerden ve uygulamanın 4H/1H kuralından tutarlı biçimde daha iyi. Ama maliyetten sonra onlar da zararda. Maliyet öncesi sonuçları ise sıfır civarında.

## Soru ve kurgu

"Fiyat çıkacak mı?" yerine şu soruldu: **"Bu 15M + 5M sinyali, stopa değmeden kâra dönecek mi?"**

- **Sinyal:** 15M ve 5M skorlarının ikisi aynı yönde ±25'i geçtiğinde (senin "al durumu"). 28 parite, 2008–2026, parite başına tek pozisyon. Toplam 957.865 sinyal oluştu.
- **İşlem:** Stop 15M ATR14'ün 1,5 katı. Hedef stopun 1,5 katı. En fazla 4 saat tutulur. Maliyet 1,5 pip (zor senaryo: 3 pip).
- **Model girdileri:** 5M/15M/1H/4H skorları, stop büyüklüğü ve maliyet oranı, oynaklık oranları, RSI, Bollinger konumu, 1H EMA200 uzaklığı, saat, gün ve NFP/FOMC yakınlığı. Hepsi giriş anında bilinen bilgiler.
- **Bölünme:** Eğitim 2008–2018 (557 bin sinyal). Doğrulama 2019–2022 (208 bin): model ve eşik burada seçildi. Son test 2023–Eylül 2026 (193 bin), tek sefer.
- **Seçilen:** hist_boost; doğrulamada en yüksek olasılıklı %10'luk dilim. Eşik 0,411 olarak donduruldu.
- **Veri eki:** Birçok çapraz paritede yerel 1H/4H arşivi eksik çıktı (GBPCHF'de beklenen mumların sadece %60'ı var). Bu yüzden 1H/4H mumları tam olan 15M arşivinden üretildi. Bu, doğrulama ve son test sonuçlarına bakılmadan önce yapıldı (data_amendment.json).

## Son test sonuçları (2023 – Eylül 2026)

| Grup | İşlem | Kazanma (1,5 pip) | Ort. R, maliyetsiz | Ort. R, 1,5 pip | Ort. R, 3 pip |
|---|---:|---:|---:|---:|---:|
| Modelin seçtikleri | 16.191 | %43,0 | +0,01 | **−0,08** | −0,16 |
| Bütün sinyaller | 192.726 | %39,0 | −0,05 | −0,22 | −0,38 |
| Uygulamanın 4H/1H kuralı | 36.088 | %38,5 | −0,06 | −0,23 | −0,40 |

Gün bazında ortalamalar, 1,5 pip maliyette:
- Modelin seçtikleri: −0,10 R, %95 GA [−0,12, −0,07]
- Bütün sinyaller: −0,24 R
- Uygulamanın kuralı: −0,24 R

Modelin seçtikleri her yıl diğerlerinden iyi:

| Yıl | Modelin seçtikleri | Bütün sinyaller |
|---|---:|---:|
| 2023 | −0,06 | −0,19 |
| 2024 | −0,08 | −0,22 |
| 2025 | −0,10 | −0,22 |
| 2026 | −0,06 | −0,24 |

| Ölçüt | Sonuç |
|---|---|
| Ortalama > 0 ve güven aralığı sıfırın üstünde | **Geçmedi** |
| Bütün sinyallerden iyi | Geçti |
| Uygulamanın 4H/1H kuralından iyi | Geçti |
| 3 pip maliyette artı | **Geçmedi** |
| En az 300 işlem | Geçti |

## Değerlendirme

Aşağıdaki ayrıştırma sonuçlar görüldükten sonra yapıldı; sadece açıklama amaçlıdır.

- **15M + 5M uyumu tek başına yön bilgisi taşımıyor.** Bütün sinyaller maliyet öncesi bile −0,05 R. Kazanma oranı %39; 1,5 R hedefli bir işlemin maliyetsiz başabaş noktası %40.
- **Uygulamanın 4H/1H kuralı bu sinyali iyileştirmiyor.** Doğrulamada da son testte de bütün sinyallerden biraz daha kötü.
- **Modelin kazancının yarısı maliyetten geliyor.** Seçtiği işlemlerde medyan stop 19,7 pip, bütün sinyallerde 10,4 pip. 1,5 pip maliyet, riske edilen tutarın yaklaşık %14'ü yerine %8'i oluyor. Diğer yarısı yönden: maliyet öncesi −0,05 R'den +0,01 R'ye çıkıyor.
- **Model en çok saat ve günün kendisine bakıyor.** Seçtiği işlemlerin çoğu 13–17 UTC arasında (Londra–New York çakışması, TR saatiyle 16–20). En sık seçtiği pariteler GBP ve JPY'liler; yani daha hareketli piyasalar.
- **Maliyet öncesi sıfır, maliyet sonrası eksi.** 0,5 pip maliyette bile modelin seçtikleri −0,02 R.

**Pratik sonuç:** Bu sinyal ailesinde yön avantajı yok. Kaybın büyük kısmı, dar stoplu işlemlerde maliyetin payının büyük olmasından geliyor. Hareketli saatlerde ve pariteler de daha geniş stopla işlem yapmak kaybı küçültüyor ama artıya çevirmiyor.

## Sınırlar

- Sabit maliyet kullanıldı. Gerçek spread gece ve haber anında daha yüksek; bu, sonuçları daha kötüleştirir.
- Arşiv fiyatları spread içermiyor ve bid/ask ayrımı yok.
- Sinyal senin gerçek işlem kararlarının mekanik bir özeti. Hangi işleme girmediğin ve çıkışı nasıl yönettiğin modelde yok.
- Pariteler ortak para birimleri paylaşıyor; bu yüzden istatistik birimi olarak gün alındı.
- Model eğitim döneminde (2008–2018) LONG ile SHORT arasında fark öğrenmiş ("side" girdisi önemli). Bu, o dönemin piyasa rejimine ait olabilir.

## Tekrar çalıştırma

`python -X utf8 research_meta_label.py build`, `select`, `final` (bu sırayla). Yerel data/historical_5m ve _15m arşivleri gerekir. dataset.csv.gz büyük olduğu için git'e eklenmez. Final aşaması sonuçlar varken tekrar çalışmaz. selected.json, validation.csv, results.json, final_by_year.csv ve final_kept_by_symbol.csv bütün sayıları içerir.
