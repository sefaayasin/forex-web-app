# 15 dakikalık girişler: inceleme ve otomatik öğrenme

8 Ekim 2026. Mevcut motor ve yeni öğrenme deneyi ayrı ayrı test edildi.
**Maliyet sonrası tutarlı tahmin üstünlüğü bulunamadı.** Öğrenme altyapısı
çalışıyor; yeni model mevcut işlem kurallarının yerine geçirilmedi.

## Mevcut motor: hesaplama düzeltmesinden sonra

8 Ekim 2026: canlı/backtest puanı birleştirildi, Wilder başlangıcı düzeltildi
ve 365 günlük testin üst zaman dilimlerinin yalnız 90–120 günü içermesi giderildi.
Önceki 155 işlemlik tablo tam filtre geçmişi sağlamıyordu ve geçersizdir.
Aşağıdaki tablo son kodla yeniden hesaplandı.

`audit_current_entry_engine.py`: gerçek run_backtest, 15M Düzeltme + Tepki,
365 gün, tüm gün, 1,5 pip masraf; ayrı parite hesaplarında 10.000 USD;
risk %0,5; RR 1,5; hibrit stop; 16 mum bekleme; 24 mum azami tutma; 1 R başa baş.
1H/4H, 15M arşivinden üretilir; filtre geçmişi 385/425 gün istenir.

| Parite | İşlem | Ortalama net R | Net USD | En büyük düşüş |
|---|---:|---:|---:|---:|
| EURUSD | 14 | +0.154 | +106.84 | %1.14 |
| GBPUSD | 12 | -0.360 | -214.33 | %2.14 |
| USDJPY | 17 | -0.562 | -468.58 | %4.69 |
| AUDUSD | 13 | -0.189 | -123.77 | %1.94 |
| USDCAD | 23 | -0.348 | -394.81 | %5.04 |
| USDCHF | 16 | -0.537 | -422.40 | %4.80 |
| NZDUSD | 14 | -0.227 | -158.88 | %2.09 |

Toplam 109 işlem. Her parite 60 işlemlik edge örneklem şartının
altında. Tek pozitif EURUSD; yeterli kanıt sayılmaz. Ayrı hesapların toplamı
bir portföy getirisi değildir. Ham sonuçlar current_engine_audit.json içindedir.
Yeni 30/100 hedef araştırması: [rapor](../cash_target/REPORT.md).

## Öğrenen giriş deneyi

`forex_adaptive.py`, saatlik yön tahmini yerine **15M teknik giriş adayının
maliyet sonrası kazanma ihtimalini** öğrenir. 2008–11 Eylül 2026 yerel arşivi
kullanılır. 2023–2026 yürüyen dönem kontrolünde 29.867 aday, 105 parite/dönem
ve 840 model seçimi eğitimi değerlendirildi. Son yedi model için ayrıca 56
aday eğitim vardır. Bu bir sınırlı otomatik model/indikatör aramasıdır;
internette sürekli araştırma veya kendi kaynak kodunu değiştirme işlemi değildir.

Karşılaştırılan gruplar:

- **price:** getiriler, geçmiş oynaklık oranı, ATR, mum gövdesi/aralığı, saat,
  yön ve maliyet/stop oranı.
- **technical:** bunlara EMA20/50/200, RSI, MACD, Bollinger ve kapanmış
  15M/1H/4H teknik skorları eklenir.
- **extended:** ayrıca ADX, DI farkı, Stochastic, Donchian konumu ve hareket
  verimliliği eklenir.

Her üç ayda bir, yalnız geçmişte kalmış verilerle:

1. Üç veya sekiz yıllık geçmişte lojistik model; geniş grupta ayrıca histogram
   boosting eğitilir. Sekiz aday karşılaştırılır.
2. Son bir yılın ilk altı ayı model seçer; ölçüt Brier kaybıdır.
3. Son altı ay ayrı sigmoid kalibrasyonunu ve %10/%20/%30/%50/%100 kapsam
   eşiklerini öğrenir. Eşik seçimi bu geliştirme verisinde ortalama net R kullanır.
4. En az 50 geliştirme işlemiyle hem 1,5 hem 3 pip maliyette pozitif sonuç
   bulunmazsa model giriş seçmez. Bu şart istatistiksel canlı işlem onayı değildir.
5. Sonraki üç ay, model/eşik değiştirilmeden değerlendirilir.

Lojistik modelin katsayıları ölçeklendirilmiş girdilerden öğrenilir ve son
kalibrasyon katsayısıyla düzeltilir. Ekrandaki yüzde, mutlak katsayıların payıdır;
başarı ihtimali veya bir indikatörün bağımsız katkısı değildir. Birbiriyle ilişkili
indikatörler katsayıları paylaşabilir. Ağaç modeli için tek bir sabit yüzde yoktur.
Geçmiş dönemlerin ağırlıkları da uygulamadaki dönem seçicisinden görülebilir.

| Parite | Seçilen işlem | Seçilenlerde net R, 1,5 pip | Net R, 3 pip | Bütün adaylarda net R |
|---|---:|---:|---:|---:|
| EURUSD | 0 | — | — | -0,204 |
| GBPUSD | 47 | -0,013 | -0,128 | -0,170 |
| USDJPY | 39 | -0,295 | -0,357 | -0,136 |
| AUDUSD | 0 | — | — | -0,219 |
| USDCAD | 0 | — | — | -0,224 |
| USDCHF | 41 | -0,036 | -0,185 | -0,199 |
| NZDUSD | 0 | — | — | -0,232 |

Toplam 127 seçilmiş işlem de pozitif ekonomik kanıt vermedi. Son güncellemede
yedi paritenin hepsinin öğrenilmiş giriş eşiği kapalıdır (`threshold: null`).
Brier referansı, ilgili dönemin eğitim kazanma oranıyla sabit tahmindir.
Yedi paritenin altısında modelin Brier kaybı referanstan kötüdür; USDCAD farkı
çok küçüktür. Kârlılık şartı olmadan zorunlu seçim yapan teşhis filtresi de
yedi paritenin tamamında zarardadır. Yeni indikatörlerin eklenmesi tek başına
güvenilir bir strateji oluşturmadı.

## Verinin ve testin sınırları

- Giriş adayı, uygulamanın vektörel yaklaşık 15M skorunun ±25 yön durumuna
  geçmesidir. Tam TREND/RANGE motorunun yapı/RSI/MACD kapıları bu aday üretimine
  uygulanmaz. Dolayısıyla yukarıdaki iki tablonun işlemleri ve çıkışları farklıdır;
  yeni modelin mevcut motora ek katkısı kanıtlanmış sayılmaz.
- Giriş, karar mumunun kapanışındaki sonraki 5M açılışıdır. Stop 15M ATR14'ün
  1,5 katı, hedef stopun 1,5 katı, azami süre dört saat. Aynı 5M mumda iki seviye
  de görülürse stop önce kabul edilir. Ters gap açılıştan, olumlu hedef gap'i
  hedef fiyatından gerçekleşir. Bunlar gerçek bid/ask gerçekleşmesi değildir.
- Bütün adaylar için dört saatlik 48 ardışık 5M mum aranır. Eksik yollar ve
  tamamlanmamış sonuçlar atılır; fiyat uydurulmaz. Bu, tam veri olan intraday
  pencerelerle sınırlı bir testtir; hafta sonu taşıma performansı ölçülmez.
- OHLC bozuklukları kaynak dosyaya yazılmaz. Bir tikten büyük bozuk 5M mum
  içeren yollar ve bozuk 15M mum sonrası 4.001 mumluk özellik bağlamı dışlanır.
  Küçük sınır düzeltmeleri ve dışlanan satırlar her paritenin raporundadır.
- Eğitim/doğrulama/kalibrasyonda, erken stopa ulaşsa bile tüm dört saatlik
  etiket penceresi sonraki döneme geçemez. Yüksek zaman dilimleri ancak tüm
  alt mumları varsa ve kapanmışsa kullanılabilir.
- Karşılaştırmalar aynı, örtüşmeyen baz aday takvimini kullanır. Model bir
  işlemi atlarsa arada yeni aday eklenmez; filtreli canlı strateji simülasyonu
  yerine aynı fırsatlarda karşılaştırma yapılır.
- Sabit 1,5/3 pip maliyet değişken spread, swap ve gerçek broker kaymasını
  modellemez. R düşüşü sabit risk birimlerindedir, yüzde hesap düşüşü değildir.
- Günlük blok bootstrap keşif amaçlıdır; pariteler için çoklu test düzeltmesi
  uygulanmamıştır. Hiçbir rapor otomatik canlı işlem izni üretmez.
- 2023 sonrası tarih daha önceki araştırmalarda incelendi. Yeni dosya veya
  yeni algoritma bu geçmişi yeniden kör test yapmaz. İleriye dönük, o an
  kaydedilmiş tahmin ve broker gerçekleşmeleriyle ayrıca doğrulama gerekir.

## Kullanım

```powershell
python -X utf8 forex_adaptive.py
python -X utf8 forex_adaptive.py --symbols EURUSD --months 3
python -X utf8 forex_adaptive.py --watch 600
python -X utf8 audit_current_entry_engine.py
python -X utf8 -m unittest discover -q
```

ML Tahmini ekranındaki **15 dakika — otomatik öğrenme ve giriş testleri**
bölümü sonuçları, dönemleri ve ağırlıkları gösterir. Yerel arşiv varsa buradan
güncelleme yapılabilir. Cloud ortamında ham arşiv olmadığı için eğitim
butonu gösterilmez; kayıtlı raporlar okunur. Büyük veri/model dosyaları git'e
eklenmez; yerel `latest.joblib` gelecekteki araştırma entegrasyonu için saklanır.

`--watch 600`, süreç ve bilgisayar çalışırken her on dakikada yerel dosyaları
kontrol eder. Yeni fiyat arşivi veya ilgili kod değişince yeniden öğrenir;
değişiklik yoksa tekrar eğitim yapmaz. İnternetten fiyat indirmez, başlangıçta
otomatik Windows görevi oluşturmaz ve broker işlemi açmaz. `monitor.json`
son kontrolü/hataları gösterir. Süreç kapanırsa izleme de biter.

Veri eğitimin ortasında değişirse son geçerli rapor/model korunur, tekrar
denemek gerekir. Kaynak hash'leri, dönem sınırları, bütün model/eşik denemeleri
ve özellik ağırlıkları `<PARITE>/report.json` içinde kayıtlıdır.

## Doğrulama ve kaynaklar

216 test geçti. Yeni 11 test; gelecekteki fiyat ve sonuçların geçmiş özellikleri
ve öğrenilmiş ağırlıkları değiştirememesini, dönem arındırmayı, tamamlanmamış
üst mumları, sonraki açılıştan girişi, stop/gap davranışını, maliyeti ve zararlı
geliştirmede beklemeyi kontrol eder. Streamlit paneli AppTest ile açıldı;
EURUSD raporu ve tabloları hatasız görüntülendi. RSI'nin yalnız yükseliş olan
pencerelerde NaN üretmesi düzeltildi: 100; değişmeyen fiyat: nötr 50; ısınma
penceresi boş kalır.

Yöntem araştırmasında başvurulan birincil kaynaklar:

- [scikit-learn: zaman sıralı doğrulama ve gap](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [scikit-learn: veri sızıntısını önleme](https://scikit-learn.org/stable/common_pitfalls.html)
- [scikit-learn: olasılık kalibrasyonu ve Brier](https://scikit-learn.org/stable/modules/calibration.html)
- [TA-Lib: momentum indikatörleri, ADX/Stochastic/RSI](https://ta-lib.github.io/ta-lib-python/func_groups/momentum_indicators.html)
- [Bailey ve diğerleri: backtest aşırı uyumu](https://escholarship.org/uc/item/4w1110bb)
