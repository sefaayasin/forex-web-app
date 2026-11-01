# MT5 son işlemler ve erken çıkış incelemesi

1 Kasım 2026. Bağlı demo hesaptaki işlem geçmişi doğrudan okundu. Son bot
işlemleri 26–27 Ekim 2026'ya ait; yakalama anında açık pozisyon yok.
Emir gönderilmedi, bot başlatılmadı, canlı günlük veritabanı değiştirilmedi.

## Sonuç

Küçük kârın başlıca nedeni yeni politikanın küçük sabit dolar hedefidir.
Eski takip politikasında ise hedef görüldükten sonra kârın geri verilmesi de var.
ML dönüşü ve sepet kapanışı ayrı mekanizmalardır; bütün küçük kapanışları
aynı nedenle açıklamak doğru olmaz.

Sanal bakiye ile boyutlanan **39 tamamlanmış işlem: −89,21 $**.
14 pozitif, 24 negatif, bir sıfır sonuç. İlk sürümün gerçek demo bakiyesine göre
boyutlanan beş işlemi bu toplama dahil değil. Eski takip politikasının 33 işlemi
−91,49 $, yeni sabit hedef politikasının altı işlemi +2,28 $.
Farklı zamanlarda, farklı risk ve giriş koşullarıyla işlem açıldığı için bu
iki toplam karşılaştırmalı strateji testi değildir.

## Son altı işlem

Net sonuçlar giriş/çıkış komisyonu, swap ve fee içerir. Aynı paritedeki iki
USDCHF işlemi ayrı pozisyonlardır.

| Pozisyon | Yön | Süre | Planlanan net hedef | Gerçek net | Kapanış |
|---|---|---:|---:|---:|---|
| EURCHF 411105871 | SHORT | 19,1 dk | 1,23 $ | +1,39 $ | Broker TP |
| USDCHF 411132939 | SHORT | 7,1 dk | 1,23 $ | +1,22 $ | Broker TP |
| AUDUSD 411165425 | SHORT | 46,4 dk | 1,23 $ | +1,19 $ | Broker TP |
| GBPUSD 411165429 | LONG | 3,6 dk | 1,23 $ | +0,98 $ | Broker TP |
| EURJPY 411165434 | LONG | 45,6 dk | 1,23 $ | +1,30 $ | Broker TP |
| USDCHF 411169054 | SHORT | 50,1 dk | 1,24 $ | −3,80 $ | Broker SL |

Beş TP toplam +6,08 $, bir stop −3,80 $, toplam **+2,28 $**.

Bu altı işlem tek bir düzenin sonucu değildir; yan yana okunmamalıdır:

| İşlemler | ML yön eşiği | ML yenileme | Kanıt |
|---|---:|---|---|
| EURCHF, ilk USDCHF | %55 | Saatlik | `bot_cost10_20261027_171423.out.log`, `signal_hour` = 14:00 |
| GBPUSD, EURJPY, AUDUSD, ikinci USDCHF | **%52** | **5 veya 10 dk kayan** | Karar kayıtları, `signal_hour` = 14:50 |

`--target-r 0.3`, yaklaşık 410 $ sanal bakiye ve `%1` riskle dolar hedefi
yaklaşık `410 × 0.01 × 0.3 = 1.23 $` olur. Hesabın yaklaşık 98 bin dolarlık
demo bakiyesi lot hesabında kullanılmıyor. Eski `%2,5` riskten `%1` riske geçiş
de lotu ve dolar kazancını küçültmüş.

GBPUSD'de planın orta fiyatı 1,331865, gerçek alış dolumu 1,33195 ve TP dolumu
1,33235. 0,03 lotta brüt 1,20 $, iki komisyon toplam 0,22 $; net 0,98 $.
Hedefin doluma kaydırılması mevcut kodda zaten düzeltilmiş; eski işlem bu
düzeltmenin geriye dönük uygulanmış hali sayılmaz.

Son USDCHF eski tam broker stopuyla kapanmış. Mevcut koddaki yarım stopun
broker emrine yazılması bu işlemden sonra eklenmiş. Eski işlem sonucu yeni
yarım stopun başarısızlığı olarak yorumlanamaz.

## Neden kâr geri veriliyor?

**Eski takipte hedef kâr alma noktası değildi.** Hedef, 1 ATR geriden takip
başlangıcıydı; stop tabanı girişin 0,5 ATR zarar tarafındaydı. Küçük 0,3R hedef
görüldüğünde bu stop net kârı korumayabiliyor. Altı takip stopu toplam yalnız
+1,71 $, ortalama +0,285 $ üretmiş. Üç somut örnek:

| Pozisyon | Hedef görüldü mü? | Gerçek net |
|---|---|---:|
| EURCHF 410989528 | Evet | −0,55 $ |
| GBPCAD 411021220 | Evet | −0,07 $ |
| AUDJPY 411063329 | Evet | −1,87 $ |

**ML dönüşünde çıkışın güven eşiği yok.** Mevcut `manage()` yönün %50 çizgisinin
öbür tarafına geçmesini yeterli görüyor. Girişteki `%55` şartı çıkışa uygulanmıyor.
Örneğin %50,1'lik ters sinyal de kapanış isteği üretebilir. `--ml-every 5`
bu kontrolü saatlikten beş dakikalığa indiriyor; dört saat yön modeli böylece
daha sık erken kapatma üretebilir. Kayan saatlerle eğitilmiş/doğrulanmış bir
model bulunmadığından daha sık güncelleme iyileşme kanıtı değildir.
Gerçekte ML dönüşüyle kapanan iki işlem toplam −2,73 $; bütün küçük kârların
nedeni bu mekanizma değil.

**Sepet kuralı tek pozisyonun hedefini beklemiyor.** Beş pozisyon bu kuralla
toplam +24,66 $ kapandı. 27 Ekim 17:02 Türkiye saati logunda sepet +17,45 $
tepeden yaklaşık +12,18 $'a geriledikten sonra üç pozisyon birlikte kapatılıyor.
Bu kapanışların biri EURJPY'de yalnız +1,33 $; diğer pozisyonların sepeti
tetiklemesiyle açıklanıyor. Kaynak: `data/bot_logs/bot_fixed_20261027_165534.out.log`.

## Daha uzun bekleseydi?

14 paritede **1.350.334 bid/ask ticki** indirildi. Deal ve tickler aynı ham
`time_msc` ekseninde eşleştirildi; bugünkü saat farkı eski işlemlere uygulanmadı.

Aşağıdaki sütun, gerçekleşen kapanıştan tam bir saat sonraki ilk geçerli
kapatma kotasyonunun yaklaşık net değeridir. Bu süre boyunca **stop, hedef,
ML ve sepet uygulanmadığı** varsayılır. Gerçekleştirilebilir alternatif
strateji sonucu değildir.

| İşlem | Gerçek net | Çıkıştan 60 dk sonra korumasız bekleme |
|---|---:|---:|
| EURCHF 411105871 | +1,39 $ | yaklaşık +4,11 $ |
| USDCHF 411132939 | +1,22 $ | yaklaşık +0,11 $ |
| AUDUSD 411165425 | +1,19 $ | yaklaşık +2,64 $ |
| GBPUSD 411165429 | +0,98 $ | yaklaşık −2,86 $ |
| EURJPY 411165434 | +1,30 $ | yaklaşık +1,95 $ |
| USDCHF 411169054 | −3,80 $ | yaklaşık −1,63 $ |
| **Toplam** | **+2,28 $** | **yaklaşık +4,32 $** |

Bu tabloda korumasız bir saat beklemek toplamda gerçek sonuçtan **daha yüksek**
çıkıyor. Tablo altı işlemlik, stopsuz bir yoldur; ne "beklemek daha iyi" ne de
"beklemek kötü" sonucu çıkarılır. Tek söylediği, GBPUSD örneğindeki gibi sonucun
işlemden işleme ters dönebildiğidir. Genel değerlendirme için bu tablo kullanılmaz;
hedef uzaklığının etkisi 2023–2026 ölçümündedir (`data/ml/trade_plan_stats.json`).

Çapraz paritelerde fiyat hareketinin dolar karşılığı gerçek kapanıştan
türetilmiştir; kur değişimi ve alternatif yürütme fiyatı tam modellenmez.
Dört saatlik alternatif sabit hedef/yarım stop yolu `ledger.csv` içinde ayrıca
var. ML, sepet ve yeni girişler bu alternatifte yok; portföy backtesti değil.
Tick boşlukları da CSV'de kayıtlıdır; işlem pencerelerinde en büyük boşluk
yaklaşık 364 saniyedir. Bu nedenle simülasyon sonuçları yaklaşık yol denemesidir.

Resmi API zaman sözleşmesi UTC datetime kullanımını açıklar:
[MetaQuotes fiyat geçmişi belgesi](https://www.mql5.com/en/docs/python_metatrader5/mt5copyratesrange_py).
Bu brokerda gözlenen kaymış epoch ve yaz/kış saati farkı nedeniyle bu incelemede
eski deal ile eski tick doğrudan eşleştirildi; genel zaman yardımcıları değiştirilmedi.

## Kullanıcının değişiklikleri ve yapılan teknik düzeltmeler

Kullanıcının `--ml-every`, kayan saatlik mum ve sinyal dönemi değişiklikleri korundu.
Sabit TP, doluma göre seviye kaydırma ve brokerdaki yarım stop mevcut kodda zaten var.
Bu incelemede stratejinin dolar hedefi veya risk parametreleri değiştirilmedi.

Eklenen düzeltmeler:

1. Kayan saatlik mumlar yinelenen ve geçersiz mum olmadan, yalnız kapanmış 5M
   mumlardan oluşur. Yüklemenin başında kesilmiş ilk saat atılır, geçmişteki diğer
   saatler boşluklu olsa da korunur. İlk sürümdeki "her saatte 12 mum" şartı broker
   H1 mumlarını yeniden üretmediği için geri alındı: günlük 22:00 UTC arası nedeniyle
   CHF'li paritelerde 90 günde 1.558 saatin 77'si siliniyor, model olasılığı saat
   başlarında H1'den ortalama 0,19 puan ayrılıyor, yön %2,3 farklı çıkıyordu.
   **Doğrulama:** 8 parite, 48 saat, 392 saat başı anında kayan mumlar broker H1 ile
   aynı saatleri ve aynı OHLC'yi verdi (en büyük fark 0,0 pip); aynı geçmiş uzunluğunda
   model olasılığı farkı 0,0000 puan, yön farkı %0.
2. Son kayan saat ayrıca denetlenir: en az 10 farklı, geçerli ve kapanmış 5M mum
   gerekir. Brokerın olağan arası (11 mum) kabul edilir, 9 mum, eski son saat ve
   kapanmamış mum reddedilir; önceki saatin tahmini güncelmiş gibi kullanılmaz.
   Sayı kontrolü gerçek kesintiyi tamamen yakalamaz: mumlar saatin son 10 dakikasında
   kesilirse 10 mum yine geçer. Geçici başarısız ML sonucu cache'e yazılmaz; sonraki
   taramada veri yeniden denenir. Başarılı tahmin aynı dönemde cache'den gelir.
3. Brokerın reddettiği kapatma isteği artık başarılı kapanış nedeni olarak
   günlüğe yazılmaz. Neden yalnız başarılı emir yanıtında kaydedilir.
4. `--report` hedefe ulaşma hesabı sabit TP kapanışlarını da içerir; eskiden
   yalnız `trailing=1` sayıldığı için yeni TP kazançları raporda dışarıda kalıyordu.

**Ayrı kayıt sorunu:** AUDUSD 411165425, EURJPY 411165434 ve USDCHF 411169054
brokerda kapanmış, bot günlüğünde henüz kapanmamış. Son heartbeat 27 Ekim
15:03 UTC; üçü bundan sonra broker TP/SL ile kapanmış. Bot çalışmadığı için
`sync_closed()` bu sonuçları yazamamış. MT5 sonuçları araştırma CSV'sinde
tamamlandı; asıl günlük bu incelemede değiştirilmedi. `--report` hâlâ yerel
günlük raporudur, canlı MT5 eşitlemesi yapmaz.

**Son çalıştırmanın ayarı:** Son dört pozisyonun giriş güveni %52,27–53,86.
Bu, mevcut %55 varsayılanının altında, ama belirsiz değil. Karar kayıtlarındaki
BEKLE nedenleri 17:44'te "en az %55", 17:50'de "en az %52" diyor; işlemlerin
`signal_hour` değeri 14:00'den 14:50'ye geçiyor ve 17:54 girişi de 14:50 dönemine
yazılıyor. Bot bu arada %52 eşiği ve 5 ya da 10 dakikalık ML yenilemesiyle yeniden
başlatılmış. Bu dört giriş %55 kuralının uygulanmış örneği sayılmaz. Bu çalıştırmanın
çıktısı `data/bot_logs/` altında yok; 18:03'te neden durduğu bilinmiyor.

## Test

```powershell
python -X utf8 -m unittest test_forex_bot test_forex_decision test_forex_trade_plan test_forex_ml_live
```

**75 test geçti** (33'ü `test_forex_bot`). Testler sahte veriyle çalışır; piyasa kapalıyken
canlı emir denenmedi. Broker H1 karşılaştırması gerçek MT5 verisiyle ayrıca yapıldı.
