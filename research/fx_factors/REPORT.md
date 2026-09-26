# Aylık döviz stratejileri: carry, momentum, trend — sonuç

26 Eylül 2026. Sonuç: dört stratejinin hiçbiri önceden yazılan ölçütü geçmedi. Carry, 2009–2026'da bankalar arası maliyetle yılda %3,1 kazandırdı ve üç alt dönemin üçünde de artıda kaldı. Ama çoklu test düzeltmesinden sonra anlamlı değil, ve perakende swap maliyetiyle yılda %1,1'e iniyor. Momentum ve trend bu dönemde zarar etti. Uygulamaya bir şey eklenmedi.

## Kurgu

- **Para birimleri:** EUR, GBP, JPY, CHF, CAD, AUD, NZD, dolara karşı (arşivde NOK ve SEK yok).
- **Fiyat:** Her ayın son iş günü, New York saatiyle 17:00 kapanışı.
- **Faiz:** FRED/OECD 3 aylık bankalar arası faiz (aylık). EUR ve GBP serileri Ocak 2026'da bitiyor; sonraki aylarda son değer kullanıldı.
- **Getiri:** Kur değişimi + faiz farkı (tutulan ay için).
- **Stratejiler** (literatürdeki tanımlarıyla, ayar yok):
  - **Carry:** Bir önceki ayın faizine göre en yüksek faizli 2 para birimini al, en düşük 2'sini sat.
  - **Kesitsel momentum (12-1):** Son 12 ayın (en son ay hariç) en iyi 2'sini al, en kötü 2'sini sat.
  - **Trend (zaman serisi momentumu):** Her para birimini, son 12 aydaki getirisinin yönünde tut.
  - **Karma:** Üçünün eşit ağırlıklı ortalaması.
- **Tutma dönemi:** Ocak 2009 – Ağustos 2026 (212 ay). Aylık yeniden dengeleme.
- **Maliyet:**
  - Bankalar arası: işlem başına 1,5 pip.
  - Perakende: 3 pip + açık pozisyon başına yılda %1 swap farkı (tipik bir CFD hesabı için varsayım).
- **Başarı ölçütü:** Bankalar arası maliyetle ortalama > 0, tek yönlü p ≤ 0,0125 (dört strateji için Bonferroni düzeltmesi) ve üç alt dönemin en az ikisinde artı. Bunun üstüne perakende maliyetle de artı kalması.
- Kurallar sonuç hesaplanmadan önce protocol.json dosyasına yazıldı.

## Sonuçlar (veri ekinden sonra)

| Strateji | Yıllık ort. | Yıllık oynaklık | Sharpe | En büyük düşüş | En kötü ay | p |
|---|---:|---:|---:|---:|---:|---:|
| Carry | **+%3,1** | %8,2 | 0,38 | −%19 | −%7,4 | 0,046 |
| Kesitsel momentum | −%2,5 | %7,8 | −0,33 | −%55 | −%9,3 | 0,90 |
| Trend | −%1,1 | %6,1 | −0,18 | −%27 | −%7,4 | 0,77 |
| Karma | −%0,2 | %4,9 | −0,04 | −%19 | −%7,0 | 0,56 |

Perakende maliyetle yıllık ortalama: carry +%1,1 (Sharpe 0,14), momentum −%4,6, trend −%2,1, karma −%1,9.

| Alt dönem (bankalar arası, yıllık) | Carry | Kesitsel mom. | Trend | Karma |
|---|---:|---:|---:|---:|
| 2009–2013 | +%5,7 | −%3,3 | −%4,6 | −%0,7 |
| 2014–2019 | +%1,0 | −%4,2 | +%0,4 | −%1,0 |
| 2020–2026 | +%3,1 | −%0,5 | +%0,3 | +%1,0 |

| Strateji | Prim var mı? | Perakendede kullanılabilir mi? |
|---|---|---|
| Carry | Hayır (p = 0,046 > 0,0125) | Hayır |
| Kesitsel momentum | Hayır | Hayır |
| Trend | Hayır | Hayır |
| Karma | Hayır | Hayır |

## Değerlendirme

- **Carry, literatürle uyumlu tek sonuç.** Yüksek faizli para birimi, düşük faizliden yılda yaklaşık 3 puan fazla getirmiş (Lustig, Roussanov & Verdelhan yaklaşık 4,8 puan buluyor). Ancak 18 yılda, 7 para birimiyle bu fark istatistiksel olarak zar zor ayırt ediliyor.
- **Carry'nin kazancı birkaç yıla bağlı.** Aşağıdakiler sonuçlar görüldükten sonra hesaplandı:
  - 18 yılın 12'si artı.
  - Toplamın yaklaşık %40'ı krizden toparlanma yılı 2009'dan (+%22,5); 2009 hariç yıllık ortalama yaklaşık %2.
  - 2013 (−%6,8) ve 2022 (−%4,9) gibi sert kayıp yılları var.
- **Perakendede carry'yi swap maliyeti belirler.** Carry portföyünde toplam pozisyon, sermayenin 2 katı. Bu yüzden broker'ın faiz farkına eklediği her %1'lik swap farkı, yıllık getiriden 2 puan götürüyor. Başabaş swap farkı yılda yaklaşık %1,6. Birçok perakende hesapta swap farkı bundan yüksektir; stratejiye başlamadan önce broker'ın gerçek swap tablosuyla hesaplanmalı.
- **Momentum ve trend G10 dövizlerinde bu dönemde çalışmadı.** Menkhoff ve diğerleri ile Moskowitz ve diğerlerinin güçlü sonuçları büyük ölçüde 2008 öncesine ait. Bu, yayın sonrası zayıflama bulgusuyla uyumlu.
- **Pratik sonuç:** Bu dört stratejiden hiçbiri, perakende koşullarında risk almaya değecek bir getiri göstermiyor. Carry "var ama küçük ve pahalı".

## Veri eki

İlk çalıştırmada 3 ay sonu fiyatı eksikti (AUD 2022-11-30 ve 2023-08-31, NZD 2026-07-31). 15M arşivinde o günlerin çoğu yok. Bu eksiklik momentum sinyallerini 12 aya kadar boş bırakıyordu. Eksik fiyatlar aynı kuralla yerel 1H arşivinden alındı (data_amendment.json).
- İlk çalıştırmanın sonuçları results_before_data_amendment.json dosyasında.
- Carry değişmedi (+%3,1).
- Kesitsel momentum −%2,9'dan −%2,5'e, karma −%0,3'ten −%0,2'ye değişti.
- Kararların hiçbiri değişmedi.

## Sınırlar

- 7 para birimi, kesitsel sıralama için az (literatür 10–40 para birimi kullanıyor). En yüksek/en düşük 2 seçimi gürültülü.
- 3 aylık bankalar arası faiz, gerçek vadeli kur primine yakın ama birebir aynı değil. Aylık ortalama olduğu için ay içindeki faiz değişiklikleri kaba yansıyor.
- %1'lik perakende swap farkı bir varsayım; broker'dan brokera değişir.
- Değer (satın alma gücü paritesi) stratejisi test edilmedi; ülke enflasyon verisi gerekir.

## Tekrar çalıştırma

`python -X utf8 research_fx_factors.py`

data/historical_1h_from_15m ve data/historical_1h gerekir. FRED faizleri ilk çalıştırmada rates.csv dosyasına indirilir. results.json bütün istatistikleri, monthly_returns.csv her stratejinin aylık getirisini içerir.
