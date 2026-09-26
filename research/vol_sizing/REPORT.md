# Oynaklık tahminiyle lot ayarı — sonuç

26 Eylül 2026. Sonuç: iki hipotez de desteklenmedi. Uygulamanın "yüksek oynaklıkta stop daha kolay tetiklenir, lotu küçült" tavsiyesi, uygulamanın kendi ATR tabanlı stopuyla veride tersine çıktı ve metin düzeltildi.

## Sorular

- **H1:** Uygulama, 72 saatlik model yüksek oynaklık beklerken "aynı stop daha kolay tetiklenebilir; lotu küçültmeyi veya stopu genişletmeyi düşün" diyordu. ATR tabanlı stopla, model "yüksek" derken açılan işlemler gerçekten daha sık stop oluyor mu?
- **H2:** O dönemlerde riski yarıya indirmek (ortalama risk aynı kalacak şekilde) günlük dalgalanmayı ve en büyük düşüşü azaltıyor mu?

## Kurgu

- **İşlemler:** research/meta_label'daki 15M + 5M sinyalleri, 2023 – 11 Eylül 2026 (188.048 işlem). Kurallar aynı: 1,5 × ATR14(15M) stop, 1,5R hedef, en fazla 4 saat, 1,5 pip maliyet. Yeni işlem kuralı eklenmedi.
- **Tahmin:** Yayındaki 72 saatlik model. 2023 öncesiyle eğitildiği için bu dönem onun için tamamen yeni veri. Her işleme girişten önce kapanmış son saatin tahmini verildi. Tahmini olmayan 4.678 işlem çıkarıldı.
- **Rejimler:** Uygulamanın eşikleri: yüksek (olasılık ≥ 0,65), olağan (≤ 0,35), belirsiz (arası).
- **İstatistik:** Farklar için işlem günlerini yeniden örnekleyen bootstrap (aynı günün işlemleri birlikte kalır).
- Kurallar ve ölçütler sonuç hesaplanmadan önce protocol.json dosyasına yazıldı.

## Sonuçlar

| Rejim | İşlem | Pay | Stop olma | Hedefe ulaşma | Ort. net R | Medyan stop |
|---|---:|---:|---:|---:|---:|---:|
| Yüksek oynaklık | 14.781 | %8 | %49,1 | %28,4 | −0,15 | 15,5 pip |
| Belirsiz | 110.990 | %59 | %51,0 | %29,4 | −0,20 | 10,2 pip |
| Olağan / sakin | 62.277 | %33 | %51,7 | %28,9 | −0,23 | 9,9 pip |

Yüksek eksi olağan (%95 GA):
- Stop olma oranı: **−2,6 puan** [−4,0, −1,4]
- Ortalama net R: **+0,08** [+0,05, +0,11]
- Net R'nin standart sapması: −0,03 [−0,04, −0,02]

| Lot kuralı | Günlük ort. R | Günlük std | En büyük düşüş | En kötü gün |
|---|---:|---:|---:|---:|
| Sabit risk | −0,2069 | 0,2068 | −238,6 | −1,10 |
| Yüksek oynaklıkta yarım risk | −0,2081 | 0,2080 | −239,9 | −1,14 |

| Ölçüt | Sonuç |
|---|---|
| H1: "Yüksek"te stop olma oranı daha fazla (GA > 0) | **Desteklenmedi**; tersi anlamlı |
| H2: Yarım risk dalgalanmayı ve düşüşü azaltır, ortalamayı düşürmez | **Desteklenmedi**; üçü de biraz kötüleşti |

Bilgi amaçlı: Uygulamanın 4H/1H kuralına uyan alt kümede de yön aynı. Stop oranı farkı −1,4 puan [−3,8, +1,0], ortalama R −0,17'ye karşı −0,24.

## Değerlendirme

- **ATR stop oynaklığı zaten hesaba katıyor.** Model "yüksek" dediğinde medyan stop 15,5 pip, "olağan"da 9,9 pip. Risk tutarı sabit olduğu için bu dönemlerde lot kendiliğinden küçülüyor. Tavsiyedeki "aynı stop" varsayımı ATR stop için geçerli değil.
- **Yüksek oynaklıktaki işlemler daha az kaybetti.** Aşağıdaki ayrıştırma sonuçlar görüldükten sonra yapıldı:
  - +0,08 R farkın yaklaşık 0,05'i maliyetten: 1,5 pip, 15 piplik stopta riskin %12'si, 10 piplik stopta %17'si.
  - Kalan yaklaşık 0,03'ü maliyet öncesi sonuçtan: −0,03 R'ye karşı −0,06 R.
- **Kayma da daha az.** Stopun ötesinde kapanan işlemler yüksek oynaklıkta %0,6, olağan dönemde %1,4.
- **Pratik sonuç:** ATR tabanlı stop kullanan biri için modelin "yüksek oynaklık" uyarısı ek lot küçültmeyi gerektirmiyor. Sabit pip stop kullanan biri için ise yüksek oynaklık gerçekten daha kolay stop demek. Uygulamadaki metin bu ayrımı yansıtacak şekilde düzeltildi.
- İşlem akışı lot ayarından önce de zararda (ortalama −0,21 R). Bu test riskin şeklini ölçüyor, kârlılığı değil.

## Protokoldeki bir ölçüm hatası

Protokolde "1,05R'den fazla kayıp" oranı stopun ötesindeki kaymayı ölçmek için tanımlanmıştı. Ancak net R'ye maliyet dahil olduğu için, 10 piplik stoptaki her normal stop da −1,15R'ye düşüyor. Bu yüzden o ölçüm kaymayı değil maliyeti ölçüyor (yüksek %42, olağan %51). Doğru ölçü, maliyet öncesi stopun ötesine taşan kayıp oranıdır ve yukarıda verildi (%0,6'ya karşı %1,4). Bu ölçüm başarı ölçütlerinde yoktu; kararları etkilemiyor.

## Sınırlar

- Kullanıcının kendi işlemleri kayıtlı değil; burada mekanik 15M + 5M sinyali kullanıldı.
- Sabit 1,5 pip maliyet varsayıldı. Oynak dönemlerde spread genişleyebilir; bu, farkın maliyetten gelen kısmını küçültür.
- Sadece bir çarpan denendi (0,5). Başka bir çarpan denemek, sonuçlara bakılarak ayar yapmak olurdu.

## Tekrar çalıştırma

`python -X utf8 research_vol_sizing.py`

research/meta_label/dataset.csv.gz ve data/historical_1h_from_15m gerekir. results.json bütün sayıları içerir.
