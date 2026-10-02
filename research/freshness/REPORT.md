# Sinyal tazeliği ve dönüş uyarısı — sonuç

2 Ekim 2026. Sonuç: iki hipotez de geçmedi. Özet kartındaki "Tazelik" ve "Dönüş sinyali" satırları sonraki hareketi tahmin etmiyor. Daha önemlisi: 15M yönünde girmek tazelikten bağımsız olarak maliyetten önce bile hafif zarar ediyor. Satırlar kartta kaldı, ama altına bunların tahmin olmadığını söyleyen bir not eklendi.

## Soru

Kullanıcı, Özet kartı 100/100 gösterirken bir işleme girip kazandı, çıktı. Kart hâlâ 100/100 gösterdiği için tekrar girdi ve kaybetti. Radar puanı trendin ne kadar güçlü olduğunu ölçer ve hareket uzadıkça 100'de kalır. Bu yüzden karta iki satır eklendi (forex_freshness.py):

- **Tazelik:** Yön ne zamandır var ve o zamandan beri fiyat ne kadar gitti?
  - Taze: tipik 4 saatlik hareketin yarısından azı gidilmiş.
  - Olgun: yarısından fazlası gidilmiş.
  - Geç: tamamı gidilmiş ya da fiyat EMA20'den 2 ATR'den fazla uzaklaşmış.
- **Dönüş sinyali:** 15M ve 5M'de ters yöne işaretler: RSI/MACD uyumsuzluğu, M/W formasyonu, MACD momentumunun zayıflaması, RSI 70 üstü/30 altı, skorun ters dönmesi.
  - Var: 15M'de en az 2, 5M'de en az 1 işaret.
  - Tek tük: herhangi bir işaret.
  - Yok: hiç işaret yok.

Sorular:

1. **H1:** Taze anda 15M yönünde girmek, geç anda girmekten daha mı iyi sonuç veriyor?
2. **H2:** Dönüş "Var" iken girmek, "Yok" iken girmekten daha mı kötü sonuç veriyor?

## Kurgu

- **Veri:** 28 parite (EURZAR hariç), 8 Ocak 2008 – 11 Eylül 2026. 15M skoru, tazelik ve 15M uyarıları Dukascopy'nin 15 dakikalık mumlarından; 5M uyarıları, giriş ve çıkış fiyatları 5 dakikalık mumlardan hesaplandı.
- **Karar anı:** 15M skorunun (uygulamanın kendi `score_series_for_backtest` fonksiyonu) mutlak değerce en az 25 olduğu her kapanmış 15M mum. Toplam 10,3 milyon an. Bir serinin yalnızca ilk anı değil, yön gösterilen her an sayıldı. Çünkü kullanıcının sorunu, kart yön gösterdiği herhangi bir anda girmek.
- **Sonuç ölçüsü:** Karar anında açılan 5M mumunun açılışından 1 ve 4 saat sonrasına kadar, 15M yönünde pip olarak hareket. Net sonuç için ölçülen spread (New York saatine göre) + 0,7 pip komisyon düşüldü.
- **Kontrol:** Araştırma, kuralları hızlı (vektörel) olarak yeniden hesaplıyor. Her paritede 300 rastgele anda bu sonuçlar `forex_freshness` modülünün kendi fonksiyonlarıyla karşılaştırıldı. Tazelik durumu, uyarı sayıları, dönüş seviyesi ve tipik 4 saatlik hareket 28 paritenin hepsinde birebir tuttu.
- **İstatistik:** Takvim günlerini bloklar halinde yeniden örnekleyen bootstrap (2.000 tekrar). Dört test olduğu için %98,75 aralık kullanıldı (Bonferroni).
- **Geçme şartı:** Fark hem 1 hem 4 saatte sıfırın üstünde ve aralığı sıfırı içermiyor olmalı. Ayrıca 4 saatlik fark üç alt dönemin (2008–2015, 2016–2023, 2024–2026) her birinde pozitif olmalı.
- Bütün kurallar sonuç hesaplanmadan önce protocol.json dosyasına yazıldı.
- **Protokolde yazmayan iki uygulama detayı:**
  - Her paritenin ilk 500 15M mumu (yaklaşık 5 gün) göstergelerin oturması için atlandı.
  - 5M frame'leri bellek yüzünden yıl yıl, önlerinde 3.000 mumluk geçmişle hesaplandı. Uygulama da 5M'yi 5 günlük pencereyle hesapladığı için bu, canlıya daha yakın.

## Sonuçlar

**Gruplara göre ortalama** (pip, 15M yönünde; net = maliyet sonrası):

| Grup | An | Hareket 1s | Hareket 4s | Net 1s | Net 4s | Kazanma 1s | Kazanma 4s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Taze | 5.751.393 | −0,17 | −0,24 | −2,87 | −2,94 | %48,3 | %48,8 |
| Olgun | 1.510.463 | −0,16 | −0,36 | −2,82 | −3,02 | %48,1 | %48,4 |
| Geç | 3.065.252 | −0,25 | −0,37 | −2,85 | −2,97 | %47,5 | %48,0 |
| Dönüş yok | 1.803.394 | −0,27 | −0,33 | −2,94 | −3,00 | %47,9 | %48,5 |
| Tek tük işaret | 6.909.443 | −0,17 | −0,28 | −2,83 | −2,94 | %48,1 | %48,6 |
| Dönüş var | 1.614.271 | −0,22 | −0,34 | −2,89 | −3,01 | %47,7 | %48,2 |

**Hipotezler** (fark, pip; köşeli parantez %98,75 aralık):

| | 1 saat | 4 saat | Geçti mi |
|---|---|---|---|
| H1: Taze − Geç | +0,08 [−0,01, +0,18] | +0,13 [−0,17, +0,46] | Hayır |
| H1, skor ≥ 90 | +0,14 [+0,01, +0,27] | +0,29 [−0,12, +0,74] | (bilgi) |
| H2: Yok − Var | −0,05 [−0,15, +0,06] | +0,01 [−0,27, +0,31] | Hayır |
| H2, skor ≥ 90 | −0,16 [−0,31, −0,03] | +0,01 [−0,37, +0,37] | (bilgi) |

Alt dönemlerde H1'in 4 saatlik farkı üçünde de pozitif: +0,04, +0,20 ve +0,21 pip. Ama aralıkların hepsi sıfırı içeriyor. H2 dönemden döneme işaret değiştiriyor.

## Yorum

- **Tazelik:** Taze girişler geç girişlerden ortalama 0,1–0,3 pip daha iyi olabilir. Yön doğru, ama fark istatistiksel olarak sıfırdan ayırt edilemiyor. Ayrıca işlem başına ~2,7 pip olan maliyetin yanında çok küçük. Taze girişler de maliyetten sonra −2,9 pip.
- **Dönüş uyarısı:** Hiçbir bilgi taşımıyor. Skor ≥ 90 iken "Var" anları 1 saatte biraz daha *iyi* çıkıyor; beklenenin tersi.
- **Asıl sorun tazelik değil:** 15M yönünde girmek, her durumda kazanma oranını %48'e ve maliyetten sonra ortalama −3 pipe getiriyor. İlk işlemin kazanıp ikincinin kaybetmesi tazelikle açıklanmıyor. Bu, ortalaması maliyet kadar eksi olan bir dağılımın içinde iki ayrı çekiliş. Bu bulgu projenin önceki 15M/5M testleriyle örtüşüyor (FOREX_ARASTIRMA.md, bölüm 3).
- **1M test edilmedi:** Yerel 1M arşivi yok; Yahoo sadece yaklaşık 30 gün veriyor. 15M ve 5M bilgi taşımadığı için 1M'den farklı bir sonuç beklemek için de bir sebep yok.

## Uygulamaya

Protokolde önceden yazılan "geçmezse" adımı uygulandı. Satırlar kartta kaldı, çünkü fiyatın nerede olduğunu doğru anlatıyorlar. Ancak altlarına tahmin olmadıklarını ve bu testin sonucunu söyleyen bir not eklendi. Satırları kaldırmak kullanıcının kararına bırakıldı.

## Dosyalar

- protocol.json — önceden yazılan kurallar
- results.json — bütün sayılar
- ../../research_freshness.py — `build` ve `evaluate` aşamaları
- ../../test_research_freshness.py — vektörel kuralların modülle aynı olduğunu sınayan testler
