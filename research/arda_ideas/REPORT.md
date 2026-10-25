# Arda'nın projesinden alınabilecek fikirler: görülmemiş veride test

25 Ekim 2026. Arkadaşın MT5 projesindeki (`arda_forex/forex-bot-main`) kurallar
ve ekran fikirleri, bizim uygulamaya eklenmeden önce bizim protokolümüzle sınandı.
Kurallar kopyalandı, hiçbiri yeniden ayarlanmadı. Ölçütler sonuçtan önce
`protocol.json` dosyasına yazıldı. **Yedi fikrin hiçbiri ölçütü geçmedi.**
Yeni sinyal eklenmedi. Haber takvimine ve İşlem Planı kartına, ölçülen
sınırlarıyla birlikte iki küçük bilgi eklendi (en altta).

## Neden yeniden test

- Arda'nın ayarları 2025 – Ekim 2026 arasındaki 21 aylık XM verisine bakılarak
  seçilmiş. Kendi notunda da "ayrı test dönemi yok" diyor.
- Bizim arşivde aynı kurallar Arda'nın hiç görmediği yıllarda çalıştırılabilir:
  dövizde 2009–2024, altında 2020–2024.
- Mumlar XM'in saatine göre kuruldu (New York + 7 saat). Böylece H4 ve günlük
  mumlar Arda'nın grafiğiyle aynı saatlerde kapanıyor.
- Maliyet: girişteki New York saatinin ölçülen Dukascopy spread'i + 0,7 pip
  komisyon, bir kez.

## Özet

| # | Fikir | Arda'nın döneminde | Görülmemiş dönemde | Sonuç |
|---|---|---|---|---|
| H1 | USDJPY H4 EMA20 geri çekilmesi | 112 işlem, +0,28 R, PF 1,49 | 1.163 işlem, −0,005 R, PF ~1,0 | Geçmedi |
| H1b | Aynı kural 28 paritede | 3.723 işlem, −0,23 R | 33.796 işlem, −0,07 R; paritelerin %18'i artı | Geçmedi |
| H2 | Altın M15 SCALP / GÜÇLÜ TREND / TREND | PF 1,39–2,70 (2026) | Spread sıfırken bile −0,03 … +0,01 R | Geçmedi |
| H3 | Günlük > H4 > H1 > M30 yön merdiveni | — | Dördü uyumluyken 4 saatte −0,3 pip (maliyetten önce), −3 pip net | Geçmedi |
| H4 | SERT HAREKET uyarısı (M15 gövdesi ≥ 3 ATR) | — | Sonraki 4 saat normalin 1,2 katı (ölçüt 1,5); yön yok | Geçmedi |
| H5 | "Bu haber X kezden Y'sinde düşürdü" satırı | — | %62–70 isabet; "iyi haber = para yükselir" kuralı da aynı | Geçmedi |
| H7 | 1R'de yarısını kapat, stop girişe | — | Ortalama R düştü, düşüş büyüdü | Geçmedi |
| H6 | En küçük lot riskin 3 katını aşarsa işlem önerme | — | Test gerekmez; USDJPY'de 1.370 işlemin 1'ini eler | Bilgi |

## H1: USDJPY H4 EMA20 geri çekilmesi

**Kural (gösterge v1.000, `USDJPY_Sinyal.mq5`):**
- Yön H4 EMA20 > EMA50'den alınır. Son kapanmış günün piyasa yapısı da aynı yönde olmalı.
- H4 mumu EMA20'ye değer, EMA20'nin yön tarafında ve yön renginde kapanır.
- Stop: son 6 mumun ucu ± 0,3 ATR. Hedef 2R. Süre 100 H4 mumu. Sinyaller arası en az 5 mum.

**Uygulama:**
- Port önce doğrulandı. 2025–2026'da 112 işlem, +0,28 R, PF 1,49 çıktı. Arda aynı dönemde 107 sinyal ve benzer PF bulmuştu.

| Dönem | İşlem | Kazanma | PF | Ortalama net R |
|---|---:|---:|---:|---:|
| Geliştirme 2009–2017 | 619 | %35 | 0,98 | −0,014 |
| Doğrulama 2018–2022 | 369 | %35 | 0,95 | −0,033 |
| Kontrol 2023–2024 | 175 | %38 | 1,13 | +0,085 |
| Arda'nın dönemi 2025–2026 | 112 | %44 | 1,49 | +0,276 |

- 2009–2024 toplamı: işlem başına −0,005 R. %95 aralık [−0,11, +0,10], p = 0,53.
- Maliyetten önce +0,038 R; medyan stop 56 pip, maliyet riskin yaklaşık %4'ü. Kuralı maliyet değil, kenarın yokluğu bitiriyor.
- AL'lar +0,04 R, SAT'lar −0,04 R.
- **28 paritede (H1b)** aynı kural −0,07 R ve paritelerin sadece %18'i artıda. 2025–2026'da 28 paritenin ortalaması da −0,23 R. USDJPY'nin son iki yıldaki başarısı paritenin o döneme özgü hareketinden geliyor gibi görünüyor.

## H2: Altın M15 kuralları

**Kurallar:** Arda'nın `research/v29_sim.py` simülatörü olduğu gibi çalıştırıldı. Sadece veri adaptörü bizim:
- **G1 SCALP hepsi:** güncel gösterge v3.000'in işlem sinyali.
- **G2 GÜÇLÜ TREND:** v2.900.
- **G3 TREND hepsi:** v2.900.

**Veri:**
- HistData M1 BID (2020 – Ekim 2026).
- Dukascopy'nin ücretsiz sunucusu dakikada bir dosyaya kısıyor. Diğer uç noktası bot doğrulaması istiyor; bu doğrulamayı aşmaya çalışmadık. Bu yüzden mum verisi HistData'dan alındı.
- Spread ölçümü için Dukascopy'den yalnız birkaç günlük BID/ASK örneği alınabildi.
- 2023 Şubat–Temmuz arası dakikaları seyrek (ayda ~20 bin, normalde ~30 bin). Gün eksiği yok.

**Adaptör kontrolü (Arda'nın dönemi, 0,30 $ spread):**

| Kural | 2026 bizde | Arda'nın notu |
|---|---|---|
| G2 GÜÇLÜ TREND | 118 işlem, PF 2,70 | 169 işlem, PF 2,67 |
| G1 SCALP | 2025 PF 2,03, Oca–Haz 2026 PF 1,62, Tem–Eki 2026 PF 1,00 | 1,94 / 1,85 / 1,78 |

Farklar, veri kaynağının farklı olmasından (XM'e karşı HistData) beklenen düzeyde.

**2020–2024 (Arda görmedi), işlem başına R:**

| Kural | İşlem | Spread 0 | Spread 0,30 $ | Ölçülen spread | AL / SAT (spread 0) | Yıllar artı (spread 0) |
|---|---:|---:|---:|---:|---|---|
| G1 SCALP hepsi | 846 | +0,007 | −0,042 | GOLD_G1 | +0,07 / −0,08 | 3 / 5 (2021, 2022 eksi) |
| G2 GÜÇLÜ TREND | 713 | −0,032 | −0,107 | GOLD_G2 | +0,04 / −0,15 | 2 / 5 |
| G3 TREND hepsi | 2.477 | +0,012 | −0,050 | GOLD_G3 | +0,06 / −0,05 | 2 / 5 |

- Spread sıfırken bile hiçbir kural sıfırdan ayrışmıyor. Sonuç spread varsayımından bağımsız.
- AL tarafı her kuralda SAT'tan iyi. Altın 2020–2024'te ~1.500'den ~2.600 dolara çıktı; yükseliş trendine göre beklenen budur, kuralın kendi katkısı değil.
- Arda'nın 2025–2026'daki güçlü sonuçları, altının çok güçlü bir trend döneminde seçilmiş ayarlardan geliyor.

## H3: Yön merdiveni

**Kurgu:**
- Arda'nın piyasa yapısı tanımı (pivot 3, son tepe/dip kırılımı) günlük, H4, H1 ve M30'da hesaplandı. Her biri son kapanmış mumundan.
- 28 parite, her saat kapanışı.
- Dördü aynı yönü saatlerin %28'inde gösteriyor.

| Dönem | An | 4 saat brüt | 4 saat net | 1 saat net |
|---|---:|---:|---:|---:|
| 2009–2017 | 429.752 | −0,51 pip | −3,17 pip | −2,85 pip |
| 2018–2022 | 239.655 | −0,36 pip | −3,01 pip | −2,79 pip |
| 2023–2026 | 168.818 | −0,27 pip | −2,93 pip | −2,78 pip |

- Uyum devam ihtimalini artırmıyor. Maliyetten önce bile hafif ters yönde.
- Sadece H4 yapısı: brüt −0,05 … −0,19 pip.
- Önceki "dört görüş aynı yönde" testiyle (research/consensus) aynı sonuç.

## H4: SERT HAREKET uyarısı

**Kural:** kapanmış M15 mumunun gövdesi, önceki 14 mumun ortalama gerçek aralığının 3 katı veya fazlası.

**Sonuç:**
- 28 paritede parite başına ayda ~8–10 kez tetikleniyor. Uyarı olarak çok sık.
- Sonraki 4 saatin aralığı, aynı saatin son 60 günlük medyanının sadece 1,2 katı (ölçüt 1,5). Mumların %32'sinde 1,5 katı aşıyor.
- **Yön:** gövde yönünde girmek maliyetten önce +0,39 / −2,55 / −0,06 pip. Net sonuç her dönemde eksi (−2,3 … −5,6 pip). Bu fark maliyetten geliyor; brütte yön bilgisi yok, tersine girmek de kazandırmaz.
- Protokoldeki "yönlü" bayrağı net sonuca baktığı için "evet" çıktı. Doğru okuma: devam ya da dönüş tahmini yok.

## H5: Haber geçmişi satırı

**Kurgu:**
- 2010–2026 ForexFactory yüksek önemli haberleri, 16.463 olay-parite.
- Sonuç: açıklamadan 60 dakika sonraki yön, haberin para birimi açısından.
- Tahmin: o serinin aynı iyi/kötü tarafındaki önceki en az 8 olayda yön ≥ %65 bir taraftaysa, sonraki olayda da o yön.

| Dönem | Tahmin | İsabet | Aynı olaylarda "iyi haber = para yükselir" |
|---|---:|---:|---:|
| 2010–2017 | 2.106 | %63,8 | %65,4 |
| 2018–2022 | 1.263 | %62,2 | %62,3 |
| 2023–2026 | 1.284 | %70,0 | %70,2 |

- Geçmiş sayımı %50'nin anlamlı üstünde. Ama genel kuraldan fazla bilgi vermiyor.
- Geçmiş, genel kurala ters bir şey söylediğinde (2010–2017'de %12) daha kötü.
- Bu 60 dakikanın çoğu ilk dakikalarda oluyor. Tüm olaylarda "iyi haber = para yükselir" açıklama anından 1 saatte %59,6 / %59,3 / %67,8 tuttu. Açıklamadan 5 dakika sonra girilseydi kalan sürede %49,3 / %48,8 / %52,0 tuttu, yani yazı tura.

## H7: Parçalı çıkış

H1 işlemlerinde (2008–2024) "1R'de yarısını kapat, stopu girişe çek, kalanı 2R / stop / süre":

| | Ortalama net R | Kazanma | En kötü ay | En büyük düşüş |
|---|---:|---:|---:|---:|
| USDJPY tam pozisyon | +0,001 | %36 | −11,4 R | 59,7 R |
| USDJPY parçalı | −0,027 | %49 | −6,9 R | 67,2 R |
| 28 parite tam | −0,061 | %34 | −118 R | 2.441 R |
| 28 parite parçalı | −0,067 | %49 | −93 R | 2.575 R |

- Kazanma oranı artıyor, en kötü ay yumuşuyor.
- Ama ortalama düşüyor ve toplam düşüş büyüyor. Ölçüt (en kötü ay ve düşüşte %30 iyileşme, ortalama > 0) karşılanmadı.
- Önceki bulguyla aynı: çıkış yönetimi kâr üretmez.

## H6: Lot koruması

- Kural: en küçük lotla (0,01) stop zararı, planlanan riskin (1.000 $ × %1,5) 3 katını aşıyorsa işlem önerilmez.
- Bu bir risk hesabı; kâr iddiası yok, test gerekmez.
- USDJPY H4 işlemlerinde 0,01 lotun stop zararı medyan 5 $; 1.370 işlemin yalnız 1'inde kural devreye girerdi.
- Altında Arda'nın simülatörü kuralı zaten içeriyor.

## Uygulamaya yansıyan

Hiçbir fikir yeni sinyal olarak eklenmedi. Var olan iki ekran, ölçülen sınırlarıyla birlikte küçük eklerle güncellendi:
- **Haber takvimi:** açıklanan değerin altına not eklendi. Not, açıklamadan sonraki 1 saatte para biriminin %59–68 o yöne gittiğini, ama 5 dakika sonra girilince bunun %49–52'ye indiğini söylüyor (H5).
- **Özet → İşlem Planı:** hedef 1R'den uzak ve lot ≥ 0,02 ise isteğe bağlı satır eklendi: "1R'de yarısını kapat, stopu girişe çek". Satır, kazanan oranını artırıp kötü ayı yumuşattığını ama kâr artırmadığını açıkça yazıyor (H7).

Diğer notlar:
- Arda'nın, kendi verisinde de kaybettiğini bulduğu fikirler (TJR, arz-talep, OB/FVG, likidite süpürmesi, EMA kesişimi, M5 scalp, grid) bizde zaten yoktu. Bu testten sonra da eklenmeyecek.
- Ders: 21 aylık tek dönemde seçilen kural, aynı enstrümanın başka yıllarında tutmuyor. Arda'nın 2025–2026'da gördüğü sonuçları bizim portumuz da üretiyor; fark verinin dönemi.

## Dosyalar

- `protocol.json`: kurallar ve ölçütler (sonuçtan önce).
- `fx_results.json`, `ladder_results.json`, `shock_results.json`, `news_results.json`, `gold_results.json`.
- Kod: `research_arda_ideas.py`; veri: `download_gold_history.py`; testler: `test_research_arda_ideas.py`.
- Tekrar: `python -X utf8 research_arda_ideas.py fx|ladder|shock|news|gold` (yerel arşiv gerekir).
