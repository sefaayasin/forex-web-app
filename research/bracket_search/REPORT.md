# 30 dolar hedef / 100 dolar stop: yeni aramalar (beş aile + 30 indikatör + altın)

9 Ekim 2026. Önceki 30/100 çalışmaları 07–17 UTC seansında 15 dakikalık
girişleri denedi. Bu turda o alanın dışında kalan beş aile sınandı.
**Hiçbiri önceden yazılmış ölçütleri geçmedi.** Uygulamaya yeni bir AL/SAT
sinyali eklenmedi.

## Neden bu beş aile

- 0,3:1 hedef/stop oranında rastgele fiyatta hedefe ulaşma oranı
  100 / 130 = **%76,9**'dur. Yani yüksek isabet tek başına avantaj değildir.
  Kâr ancak gerçek bir yön eğilimi veya ortalamaya dönüşle gelir.
- Önceki testlerde 15M stopta masraf riskin %15–25'iydi. Bu turda masrafın
  riske oranı düşük olan alanlar (günlük, haftalık) ve daha önce bakılmamış
  saatler (gece seansı) seçildi.
- Gece fikrinin dayanağı: döviz getirilerinde dönüş en çok 15 dakikalık ufukta
  görülür; New York kapanışı sonrası seans ince ve yataydır.

## Ortak kurallar

- Hedef 30 dolar net, planlanan stop 100 dolar. İkisi de masraf düşülerek ölçeklenir.
- Karar kapanmış mumdan; giriş sonraki 5M açılışından. Aynı mumda iki seviyeye
  temas olursa önce stop. Ters boşlukta açılış fiyatından çıkış.
- Masraf: kararın New York saatindeki ölçülen spread + 0,7 pip komisyon.
  Stres senaryosu iki katı. Haber testinde spread ayrıca 1,5 kat.
- Günlük testte gecelik swap, önceki ayın 3 aylık faiz farkından eksi yıllık
  %1 broker payıyla hesaplandı; çarşamba üç gece sayıldı.
- Dönemler: geliştirme 2009–2017 (haberde 2010–2017), doğrulama 2018–2022,
  kontrol 2023 sonrası.
- Ölçüt: geliştirmede en az 100 işlem ve stres masrafıyla pozitif, doğrulamada
  en az 50 işlem ve pozitif. Kontrol dönemi yalnız raporlanır.
- Kurallar ve ayar ızgarası, sonuç hesaplanmadan her klasördeki
  `protocol.json` dosyasına yazıldı.

## Sonuçlar

| Aile | Ayar | Geliştirmeyi geçen | Doğrulamayı da geçen | Kontrolde pozitif | Masraf / risk | Kontrol işlemi | Kontrol net $/işlem |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gece seansı ortalamaya dönüş (15M, 28 parite) | 336 | 0 | 0 | 0 | %8,1 | 306.240 | −11,36 |
| Günlük trend, dönüş, carry (28 parite) | 1.120 | 105 | 23 | 3 | %1,7 | 289.223 | −2,59 |
| Hafta sonu boşluğu kapanışı (28 parite) | 224 | 10 | 0 | 0 | %3,7 | 8.836 | +2,85 |
| COT spekülatör pozisyonu (6 parite) | 72 | 24 | 11 | 2 | %0,6 | 12.445 | −1,87 |
| Haber sürprizi sonrası devam (7 parite) | 56 | 0 | 0 | 0 | %5,8 | 9.382 | −8,67 |

Toplam **1.808 ayar**. İşlem sayıları ayarların toplamıdır; aynı fırsat birden
fazla ayarda sayılır ve pariteler korelasyonludur.

### Gece seansı

- Maliyetten **önce** gerçek bir dönüş var: geliştirmede işlem başına +0,034 R.
  EURCHF, GBPCAD, GBPCHF ve EURGBP'de +0,07–0,10 R.
- Ama yıllar içinde eriyor: doğrulamada +0,016 R, kontrolde −0,003 R.
- Hiçbir paritede maliyet öncesi kazanç masrafın riske oranını geçmedi.
  Piyasa bu dönüşü artık daha hızlı fiyatlıyor.

### Günlük ölçek

- Masraf riskin %1,7'sine indi, ama maliyet öncesi kazanç da sıfır (+0,001 R).
  Hedef oranı rastgele fiyatın %76,9'una çok yakın.
- 1.120 ayardan 105'i geliştirmeyi, 23'ü doğrulamayı geçti. Bu 23'ün yalnız
  3'ü kontrolde pozitif; şans eseri beklenenden bile az. Seçilenler gürültü.
- Günlük ölçekte işlem başına kayıp 15M ve gece seansına göre en küçük
  (−2 ile −3 $). Yani en az kaybettiren ölçek bu, ama kazandırmıyor.

### Hafta sonu boşluğu

- Kontrol döneminde (2023+) +2,85 $/işlem görünüyor. Ama aynı kurallar
  2018–2022'de −9,23 $/işlem. Dönemler arasında işaret değişen sonuç
  güvenilir sayılmaz.

### COT

- GBPUSD'de "aşırı spekülatör pozisyonuna karşı" kuralı üç dönemde de pozitif
  (+0,87 / +4,09 / +5,38 $; kontrolde 104 işlem). 72 ayardan biri ve kontrol
  ortalaması sıfırdan istatistiksel olarak ayrılmıyor: günlük %95 aralık
  [−2,80; +13,76] $, p ≈ 0,10; 72 ayar için düzeltilince anlamsız. Swap bu testte
  düşülmedi. Bulgu sayılmaz.

### Haber sürprizi

- ForexFactory'nin 2010–2026 arşivi indirildi: 7.242 yüksek etkili sürpriz.
  Yön, ForexFactory'nin kendi "para birimi için iyi/kötü" etiketinden alındı.
- Açıklamadan 5 ve 15 dakika sonra girildiğinde maliyet öncesi bile kazanç yok
  (−0,007 R). Sürpriz ilk dakikalarda fiyatlanıyor; bu, Andersen ve
  diğerlerinin (2003) bulgusuyla uyumlu.

## Ek tur: projede hiç denenmemiş 30 indikatör ve altın

Önceki turlar RSI, MACD, Bollinger, EMA, Ichimoku, Stochastic, Donchian,
Keltner, ADX, Fisher, Roofing, FRAMA, KAMA, Supertrend, ConnorsRSI ve Laguerre
RSI'yi yalnız 15M'de denemişti. Bu turda (`research_indicator_zoo.py`):

- **Hacim** (Dukascopy ECN hacmi; fiyattan farklı bir girdi): OBV teyitli
  kırılım, OBV uyumsuzluğu, MFI, Chaikin para akışı, günlük VWAP sapması,
  hacim patlaması sonrası dönüş.
- **Pivotlar:** klasik pivot S1/R1 dönüşü, Camarilla H3/L3 dönüşü ve H4/L4 kırılımı.
- **Likidite / SMC:** önceki gün tepe/dip süpürmesi, Londra açılışında Asya
  aralığı süpürmesi, adil değer boşluğu (FVG) yeniden testi.
- **Mum formasyonları:** 20 mumluk uçta yutan mum ve pin bar, iç bar kırılımı, NR7 kırılımı.
- **Diğer trend/osilatörler:** Parabolic SAR, Aroon, Vortex, TRIX, TSI,
  Heikin Ashi, Hull MA, Elder Ray, CCI, Williams %R, Ultimate Oscillator,
  Choppiness filtreli bant dönüşü, otokorelasyon rejimi.
- 10 parite; **15M, 1H ve 4H** karar mumu; stop 1,5 / 3 ATR; tutma 4 saat /
  1 gün / 5 gün. Gecelik swap payı yıllık %1. Ölçüt kural düzeyinde: 10
  paritenin toplamı stres masrafıyla pozitif ve en az 6 paritede pozitif.
  Kurallar geleceğe bakmama testinden geçti (`test_research_indicator_zoo.py`).

| Sonuç | Değer |
|---|---|
| Kural × zaman dilimi × stop | 178 (parite bazında 1.780) |
| Toplam ölçütü geliştirmede geçen | **0** |
| En iyi geliştirme sonucu | 4H yutan mum: +1,20 $ → doğrulama −3,46 $ → kontrol −0,63 $ |
| Parite bazında geliştirme + doğrulamayı geçen | 7 (hepsi 4H); 6'sı kontrolde pozitif |
| En iyi tekil aday | GBPUSD 4H Ultimate Oscillator: kontrolde +9,89 $, p = 0,029; 7 aday için düzeltilince p ≈ 0,20 |

Parite bazındaki 7 adayın ortak teması "4H uçta dönüş mumu / osilatör"dü.
Ama aynı kurallar 10 paritenin toplamında doğrulama döneminde eksi
(yutan mum −3,46 $, MFI −4,77 $). Tema genel değil; tek paritelerdeki başarı
gürültüyle uyumlu. 1H ve 15M'de geliştirmeyi geçen tek bir ayar bile yok.

**Altın** (`research_gold_daily.py`): Yahoo'nun 2000–2026 günlük COMEX altın
vadelisi; aynı 30 indikatör ve günlük trend kuralları, 148 ayar. Geliştirme
2001–2012 (yükseliş), doğrulama 2013–2018 (düşüş/yatay), kontrol 2019+.
Hiçbir ayar geçmedi. Yalnız günlük mum olduğundan aynı gün hedef ve stop
görüldüğünde sıra bilinmiyor (işlemlerin %7,4'ü). En iyimser varsayımda bile
(önce hedef) geçen ayar yok. "Her zaman al" referansı 2019+ yükselişinde bile
−1,5 ile −5 $/işlem: 0,3:1 yapısında finansman ve masraf, fiyat eğiliminden büyük.
Altın sonucu forexten daha kaba bir yaklaşımdır; Dukascopy'nin gün içi altın
verisi bugün indirilemedi.

## Ne anlama geliyor

- "Girdim, en kötü 30 dolar kazanıp çıktım" garantisi veren bir kural
  bulunamadı. 30/100 yapısında stop yaklaşık her dört işlemden birinde gelir
  ve tek stop üç hedefin kazancını siler. Çıkış yönetimi (başa baş stop,
  kâr kilitleme) kâr üretmez; yalnız kazanç/kayıp dağılımını değiştirir.
- Önceki 11 çalışma ve bu beş aile aynı sonucu veriyor: perakende masrafıyla,
  yalnız fiyat, takvim veya COT verisinden kârlı kısa vadeli yön bulunamadı.
- Aramayı daha da uzatmak, sonunda şans eseri "pozitif" bir ayar bulur.
  Günlük tablodaki 23 aday bunun örneği. Bu yüzden arama burada durduruldu.

## Kalan dürüst seçenekler

- **Masrafı ölçmek:** Gece dönüşü maliyet öncesi vardı. Gerçek hesap masrafı
  ölçülmeden daha düşük masraf varsayılmamalıdır.
- **İşlem yapılacaksa:** Kaybı en aza indiren yapı günlük ATR stoplu işlem;
  15M stopta masraf payı 5–10 kat daha büyük.

## Çalıştırma

```powershell
python -X utf8 research_night_bracket.py
python -X utf8 research_daily_bracket.py
python -X utf8 research_weekend_gap.py
python -X utf8 research_cot_bracket.py
python -X utf8 download_forexfactory_history.py
python -X utf8 research_news_drift.py
python -X utf8 research_indicator_zoo.py
python -X utf8 research_gold_daily.py
python -X utf8 research_bracket_summary.py
python -X utf8 -m unittest test_research_bracket_core test_research_indicator_zoo
```

Simülatör: `research_bracket_core.py` (numba). Testleri: aynı mumda stop
önceliği, ters boşluk, zaman çıkışı, tek pozisyon, eksik giriş mumu ve bozuk
fiyat barı. Ham işlem dosyaları yerelde tutulur (`.gitignore`).

## Sınırlar

- 2023+ verisi bu aileler için ilk kez kullanıldı, ama aynı arşiv önceki
  araştırmalarda incelendi.
- Projede hiç kullanılmamış 2003–2007 verisi indirilmek istendi; Dukascopy
  bugün istekleri reddetti (HTTP 202). Bu doğrulama yapılamadı.
- 2026 masraf profili eski fiyatlara uygulandı. Gece ve haber anı masrafları
  gerçek hesapta daha yüksek olabilir.
- Dolarlar sabit risk ölçeğidir; lot adımı, teminat ve kayma modellenmedi.
  Boşluklar planlanan 100 doları aşabilir (en kötü işlem −200 $ civarı).

Kaynaklar: [Andersen, Bollerslev, Diebold & Vega (2003)](https://www.nber.org/papers/w8959),
[Breedon & Ranaldo, gün içi döviz örüntüleri](https://www.snb.ch/en/publications/research/working-papers/2011/working_paper_2011_04),
[Krohn, Mueller & Whelan](https://www.bankofcanada.ca/wp-content/uploads/2021/10/swp2021-48.pdf),
[Moskowitz, Ooi & Pedersen (2012)](https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum),
[Lustig, Roussanov & Verdelhan (2011)](https://www.nber.org/papers/w14082).
