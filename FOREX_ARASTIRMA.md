# Forex: nedir, neyi tahmin edebiliriz, bu projede ne yapılabilir

26 Eylül 2026. Akademik literatür, düzenleyici verileri ve bu projede yapılan testlerin birlikte okunmasıdır. Yatırım tavsiyesi değildir.

## 1. Forex kısaca

- **Merkezi bir borsa yok.** İşlemler bankalar, piyasa yapıcılar ve elektronik platformlar arasında tezgâh üstü (OTC) yapılır. Perakende trader'ın gördüğü fiyat, kendi broker'ının fiyatıdır. Bu yüzden iki kaynak aynı saat için farklı fiyat ve farklı mum fitilleri gösterebilir; bu projede Yahoo ile Dukascopy arasında da görüldü.
- **Büyüklük:** BIS 2025 anketine göre günlük işlem hacmi 9,6 trilyon dolar. Bunun 3 trilyonu spot işlem (%31). Dolar, işlemlerin %89'unda taraflardan biri. İşlemlerin %38'i Londra'da yapılıyor.
- **Fiyatı ne hareket ettirir:**
  - Faiz farkları ve para politikası beklentileri
  - Makro verilerin beklentiden sapması (sürpriz)
  - Risk iştahı ve fonlama likiditesi
  - Kısa vadede emir akışı: kimin ne kadar alıp sattığı
- **Seanslar:** Tokyo, Londra ve New York. Hacim ve oynaklık birlikte artar, spread ters yönde hareket eder. Spread en dar Londra saatlerinde; likiditeyi en çok günün saati, sonra haftanın günü belirler (Ito & Hashimoto).
- **Perakendenin gerçeği:** AB düzenleyicisi ESMA'nın analizinde CFD hesaplarının %74–89'u zarar ediyor. Kaldıraç kısıtlanınca perakende forex trader'larının kayıpları yaklaşık %40 azalıyor. Çünkü yüksek kaldıraç, zarardaki pozisyonu kapatmayı erteletiyor (Heimer & Simon).

## 2. Literatür neyin tahmin edilebildiğini söylüyor

| Bulgu | Vade | Kanıt | Bu projede gördüğümüz |
|---|---|---|---|
| Kısa vadede yön, rastgele yürüyüşten iyi tahmin edilemiyor | Günler–aylar | Meese & Rogoff (1983); onlarca yıldır "bulmaca" olarak duruyor | Dört ayrı yoldan denendi: 15M/5M sinyali, haber anı, ML yön modeli, meta-etiketleme. Hepsinde maliyet sonrası avantaj yok |
| Haber sürprizi fiyatı anında sıçratıyor; kötü haberin etkisi daha büyük | Dakikalar | Andersen, Bollerslev, Diebold & Vega (2003) | Yönü teknik göstergeler değil sürpriz belirliyor. Haber testinde 15M/5M yönü tepkiyi bilemedi |
| Emir akışı günlük hareketin büyük kısmını açıklıyor (R² > %60) | Gün | Evans & Lyons (2002) | Bu veri perakendeye açık değil |
| Oynaklık tahmin edilebilir | Saatler–haftalar | Corsi (2009, HAR modeli) | 72 saatlik model basit tahmini geçiyor (%77,9'a karşı %71,9, 19/28 parite) |
| Oynaklığa göre pozisyon küçültmek, risk başına getiriyi artırıyor (carry dahil) | Aylar | Moreira & Muir (2017) | ATR stop bunu zaten yapıyor; üstüne model bazlı ek küçültme fayda vermedi (research/vol_sizing) |
| Carry: yüksek faizli para birimi, düşük faizliden yılda yaklaşık 4,8 puan fazla getiriyor (maliyet sonrası) | Aylar | Lustig, Roussanov & Verdelhan (2011) | 2009–2026, 7 para birimi: yılda +%3,1 (Sharpe 0,38), düzeltme sonrası anlamlı değil; perakende swapla +%1,1 (research/fx_factors) |
| Carry nadiren ama sert çöküyor; getirisi bir çeşit sigorta primi | Aylar | Brunnermeier, Nagel & Pedersen (2008) | — |
| Momentum: son dönemin kazananı ile kaybedeni arasında yılda %10'a varan fark. Maliyete duyarlı, pratikte kolay kullanılamıyor | 1–12 ay | Menkhoff, Sarno, Schmeling & Schrimpf (2012) | 2009–2026'da yılda −%2,5 (research/fx_factors) |
| Trend (zaman serisi momentumu) döviz vadelilerinde 1–12 ay sürüyor | 1–12 ay | Moskowitz, Ooi & Pedersen (2012) | 2009–2026'da yılda −%1,1 (research/fx_factors) |
| Değer (satın alma gücü paritesinden sapma); momentumla ters korelasyonlu | Yıllar | Asness, Moskowitz & Pedersen (2013) | Test edilmedi |
| Planlı FOMC günlerinde "dolar sat, diğerlerini al" stratejisinin getirisi belirgin yüksek | Gün | Mueller, Tahbaz-Salehi & Vedolin (2017) | 2008–2013'te çok güçlü (+25 bp, p=0,001). Makale sonrası 2014–2026'da +4,8 bp, anlamlı değil (p=0,13). Ayrıntı: research/fomc_day/REPORT.md |

**Özet:** Literatürdeki kalıcı bulgular ya **çok kısa** (haber sürprizine dakikalar içindeki tepki) ya da **uzun** vadeli (günlerden aylara, portföy düzeyinde carry, trend ve değer). 15 dakika–4 saat arası tek paritede yön tahmini için güçlü bir kanıt yok; bu projenin testleri de bununla örtüşüyor. Güvenilir biçimde tahmin edilebilen şey **oynaklık**. Ayrıca yayımlanan birçok avantaj, yayımlandıktan sonra zayıflamıştır; her biri maliyetle birlikte kendi verimizde yeniden test edilmelidir.

## 3. Bu projeden çıkan dersler

1. **15M/5M sinyali yön bilgisi taşımıyor.** 2008–2026 arası 28 paritede yaklaşık 960 bin sinyalde maliyet öncesi bile −0,05 R. Uygulamanın 4H/1H kuralı bunu iyileştirmiyor.
2. **Kaybı çoğunlukla maliyet belirliyor.** 10 piplik stopta 1,5 pip maliyet, riskin %15'i demek.
3. **Veri kalitesi sonuçları değiştiriyor.**
   - Yerel 1H arşivinde bütün haftalar eksikti; artık 15M'den yeniden kuruluyor.
   - Yahoo'nun saatlik fitilleri CAD/CHF çaprazlarında yaklaşık 2 kat geniş; bu, oynaklık modelini 13 paritede yanıltıyordu. Artık düzeltiliyor.
   - Yahoo EURUSD fiyatı Dukascopy'den medyan 3,6 pip farklı; pip bazlı stop ve hedefler broker fiyatıyla kontrol edilmeli.
4. **Önceden yazılmış protokol işe yarıyor.** Protokolsüz bakılsaydı birkaç test "başarılı" görünebilirdi.

## 4. Neler yapılabilir (öncelik sırasıyla)

### A. FOMC günü dolar etkisini test etmek — yapıldı, geçmedi
- **Sonuç:** Etki makalenin döneminde (2008–2013) çok güçlü, yayından sonra (2014–2026) beşte bire inmiş ve anlamlı değil. Maliyet sonrası yılda kabaca 0,2 puanlık bir getiri; perakende için anlamsız.
- **Açık kalan soru:** Açıklamadan önceki pencerede (önceki gün 16:00 → 14:00) +7,5 bp fark görüldü (p=0,03). Ama bu sonuçlar görüldükten sonra öne çıkarıldı. Ancak 2026 sonrası yeni FOMC günleriyle, önceden yazılmış bir testle doğrulanırsa değerlendirilebilir.
- Ayrıntı: research/fomc_day/REPORT.md

### B. Oynaklığa göre pozisyon büyüklüğü — yapıldı, gerek çıkmadı
- **Sonuç:** Uygulama zaten ATR'ye göre stop koyup riski sabit tutuyor; yani lot oynaklıkla kendiliğinden küçülüyor. 2023–2026'daki 188 bin 15M/5M işleminde:
  - Model "yüksek oynaklık" derken açılan işlemler daha **az** stop oldu (−2,6 puan) ve daha az kaybetti.
  - O dönemlerde lotu ayrıca yarıya indirmek sonucu iyileştirmedi.
  - Uygulamadaki "lotu küçült" tavsiyesi buna göre düzeltildi.
- **Neden literatürle çelişmiyor:** Moreira & Muir'in bulgusu, sabit pozisyonlu (oynaklığa göre ayarlanmamış) portföyler içindi. ATR stop bu ayarı zaten yapıyor.
- Ayrıntı: research/vol_sizing/REPORT.md

### C. Aylık portföy stratejileri (carry + momentum + trend) — yapıldı, geçmedi
- **Sonuç (2009–2026, 7 para birimi, aylık):**
  - Carry yılda +%3,1 (Sharpe 0,38) ve üç alt dönemde de artı. Ama çoklu test düzeltmesinden sonra anlamlı değil, ve toplamın yaklaşık %40'ı 2009 yılından.
  - Kesitsel momentum yılda −%2,5, trend −%1,1.
- **Perakende koşulları:** Carry portföyünde toplam pozisyon sermayenin 2 katı olduğu için, broker'ın her %1'lik swap farkı yıllık getiriden 2 puan götürüyor. Başabaş swap farkı yılda yaklaşık %1,6; tipik bir %1 varsayımıyla getiri yılda +%1,1'e iniyor.
- **Değer stratejisi** test edilmedi; ülke enflasyon verisi gerekir.
- Ayrıntı: research/fx_factors/REPORT.md

### D. Haber sürprizi arşivi — şimdi başlamak
- **Ne:** ForexFactory takvimi her hafta beklenti ve gerçekleşen değerlerle birlikte kaydedilir, örneğin ücretsiz bir GitHub Actions görevi ile.
- **Neden:** Haber sonrası yönü sürpriz belirliyor ama elimizde geçmiş sürpriz verisi yok.
- **Sonra:** 6–12 ay içinde "sürprizin yönü, sonraki 15–60 dakikayı tahmin ediyor mu?" sorusu test edilebilir.
- **Sınır:** Tepkinin çoğu ilk dakikalarda olur. O dakikalarda spread ve kayma en yüksek.

### E. Gerçek maliyeti ölçmek — yapıldı
- **Ölçüm:** Dukascopy'nin alış/satış fiyatlarından 28 paritenin spreadi ölçüldü (Haziran–Ağustos 2026, 10 gün, dakikalık). Ayrıca 3 NFP ve 1 FOMC anının tick verisi incelendi.
- **Asıl fark maliyetin stopa oranında:** JPY'li pariteler ve GBPUSD/EURUSD'de maliyet, tipik 15M stopun yaklaşık %7–11'i. CHF/CAD/NZD çaprazlarında %20–26.
- **Rollover:** New York 17:00'de spread 2–7 katına çıkıyor.
- **Haber anı:** Açıklamanın ilk saniyelerinde spread 5–25 katına sıçrıyor, çoğunlukla birkaç saniyede normale dönüyor.
- **Önceki testler:** Araştırmalarda kullanılan 1,5 pip, 28 paritenin 20'sinde gerçek maliyetin altındaydı; yani o testlerin sonuçları gerçekte daha da kötü olurdu.
- **Uygulamaya yansıyanlar:**
  - Maliyet ayarının varsayılanı artık ölçülen spread + komisyon.
  - Sinyal sekmesinde "maliyet planlanan kaybın %X'i" notu.
  - Tarama tablosunda şu anki spread sütunu.
- Ayrıntı: research/spreads/REPORT.md

### F. Canlı veri kaynağı — değerlendirildi, Yahoo + düzeltmeyle devam
- **Dukascopy** (modellerin eğitildiği kaynak) canlı kullanım için denendi, uygun değil:
  - Dosya başına 20–30 saniye sürüyor, sunucu sık sık 503 hatası ve zaman aşımı veriyor.
  - İçinde bulunulan ayın saatlik dosyası yok (404); gün içi veri saat saat tick dosyası gerektiriyor.
  - Sadece geçmiş analiz için kullanılabilir.
- **Yahoo + fitil düzeltmesi** yeterli çalışıyor: ML oynaklık kararının eğitim verisiyle uyumu 28/28 paritede %80'in üstünde (medyan %92).
- **Yahoo'nun eksikleri:**
  - Fiyat, broker fiyatından birkaç pip farklı olabiliyor (EURUSD'de medyan 3,6 pip). Pip bazlı stop/hedef, işlem öncesi broker ekranından kontrol edilmeli.
  - Spread bilgisi yok. Maliyet için ölçülmüş ECN spread profili kullanılıyor (E).
- **Daha iyi seçenekler (ileride):** Kendi broker'ının MetaTrader 5'i (uygulamada destek var, sadece yerelde çalışır) ya da bir broker API'si, örneğin OANDA (API anahtarı gerekir). Kullanıcı kararıyla (26 Eylül 2026) şimdilik Yahoo + düzeltmeyle devam ediliyor.

### Yapılmamasını önerdiğim şeyler
- 15M/5M'ye yeni gösterge veya filtre eklemek. Denenen her varyant maliyet sonrası zarar etti.
- Kısa vadeli yön için daha büyük ML modelleri denemek. 1.680 eğitim ve yaklaşık 960 bin sinyalle yeterince denendi.
- Sonuçlara bakarak parametre seçmek. Geçmişte iyi görünen her ayar, gelecekte aynı sonucu vermez.

**Durum:** A, B ve C tamamlandı; üçü de önceden yazılan ölçütü geçmedi. E yapıldı ve uygulamaya eklendi. F değerlendirildi; şimdilik Yahoo + düzeltmeyle devam ediliyor. Açık kalan: D (haber sürprizi arşivi; zaman istiyor, kod değil).

## Kaynaklar

- BIS Triennial Survey 2025: [bis.org/statistics/rpfx25_fx.htm](https://www.bis.org/statistics/rpfx25_fx.htm), [basın bülteni](https://www.bis.org/press/p250930.htm)
- ESMA ürün müdahalesi (CFD hesaplarının %74–89'u zararda): [esma.europa.eu](https://www.esma.europa.eu/press-news/esma-news/esma-agrees-prohibit-binary-options-and-restrict-cfds-protect-retail-investors)
- Heimer & Simon, kaldıraç kısıtı ve perakende forex: [Cleveland Fed WP 14-33](https://www.clevelandfed.org/-/media/project/clevelandfedtenant/clevelandfedsite/publications/working-papers/2014/wp-1433-can-leverage-constraints-help-investors-pdf.pdf)
- Ito & Hashimoto, gün içi mevsimsellik: [NBER w12413](https://www.nber.org/papers/w12413)
- Meese & Rogoff (1983): [ResearchGate](https://www.researchgate.net/publication/5035622_Empirical_Exchange_Rate_Models_of_the_Seventies_Are_Any_Fit_to_Survive)
- Andersen, Bollerslev, Diebold & Vega (2003): [AEA](https://www.aeaweb.org/articles?id=10.1257%2F000282803321455151)
- Evans & Lyons (2002): [NBER w7317](https://www.nber.org/papers/w7317)
- Corsi (2009), HAR modeli: [ScienceDirect rehberi](https://www.sciencedirect.com/science/article/abs/pii/S0378426621002417)
- Moreira & Muir (2017): [NBER w22208](https://www.nber.org/papers/w22208)
- Lustig, Roussanov & Verdelhan (2011): [NBER w14082](https://www.nber.org/papers/w14082)
- Brunnermeier, Nagel & Pedersen (2008): [NBER w14473](https://www.nber.org/papers/w14473)
- Menkhoff, Sarno, Schmeling & Schrimpf (2012): [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1809776)
- Moskowitz, Ooi & Pedersen (2012): [AQR](https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum)
- Asness, Moskowitz & Pedersen (2013): [AQR](https://www.aqr.com/Insights/Research/Journal-Article/Value-and-Momentum-Everywhere)
- Mueller, Tahbaz-Salehi & Vedolin (2017): [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2705818)
