# İnternetten araştırılan yeni indikatörler ve farklı bilgi kaynağı

8 Ekim 2026. Yedi yeni indikatör ailesi kodlandı ve geçmişte test edildi.
Fiyat indikatörlerinden kârlı bir aday çıkmadı. Ekonomik haberin açıklanan
değeriyle beklentisi arasındaki fark ayrıca sınandı; bazı pozitif kontrol
ortalamaları bulundu, fakat güvenilir bir canlı giriş yöntemi henüz doğrulanmadı.

## Yeni formüller ve kaynakları

| Aile | Denenen işlev | Birincil kaynak |
|---|---|---|
| Fisher Transform | Normalleştirilmiş fiyatın dönüşü ve sıfır geçişi | [Ehlers makalesi](https://c.mql5.com/forextsd/forum/3/130fish.pdf) |
| Roofing / SuperSmoother | Salınım filtresi; erken ve teyitli giriş ayrı | [Ehlers çalışması](https://c.mql5.com/forextsd/forum/122/predictiveindicators.pdf) |
| FRAMA | Fiyatın fraktal ölçüsüne uyarlanan ortalama | [Ehlers çalışması](https://c.mql5.com/forextsd/forum/208/FRAMA%20g%20Fractal%20Adaptive%20Moving%20Average.pdf) |
| KAMA | Hareket verimliliğine uyarlanan ortalama | [TA-Lib formülü](https://ta-lib.org/functions/kama.html) |
| Supertrend | ATR tabanlı takip bantları | [TradingView formülü](https://www.tradingview.com/support/solutions/43000634738-supertrend/) |
| ConnorsRSI | Kısa RSI, hareket serisi ve getiri sırası | [Geliştiricinin formülü](https://www.tradingmarkets.com/media/2012/CRHS.pdf) |
| Laguerre RSI | Dört kademeli filtreyle momentum | [Ehlers makalesinin kopyası](https://doczz.net/doc/8103163/time-warp-%E2%80%93-without-space-travel) |

Kaynaklar ve deneyde kullanılan başlangıç/değer farkları `sources.json` içinde.
Formüllerin yayımlanmış olması, forexte kârlı olduklarını göstermez. Günlük
hisse/vadeli örneklerinin 15M forexe uyarlanması yeni bir deneydir.

## Fiyat indikatörü karşılaştırması

Sekiz parite: yedi ana parite ve EURCHF. 14 giriş kuralı × iki stop genişliği
(1,5/3 ATR) × iki tutma süresi (1/4 saat) × üç planlanan zarar bütçesi
(20/50/100 dolar): **1.344 geliştirme ayarı**. Net hedef her durumda 30 dolar.
2016–2018 ve 2019–2022 dönemlerinin her birinde en az 100 işlem aranır;
en zayıf alt dönemin iki kat masraflı ortalaması en iyi olan aday seçilir.
2023 sonrası veriler seçimde kullanılmaz. Bu tarih daha önce incelendiği için
yeniden kör test olduğu iddia edilmez.

Seçilen sekiz adayın sonraki döneminde toplam **10.941 varsayımsal işlem**:

| Parite | Seçilen giriş | Bütçe | İşlem | Ort. net USD | İki kat masrafla USD |
|---|---|---:|---:|---:|---:|
| EURUSD | Connors dönüşü | 20 | 1548 | -0,91 | -1,88 |
| GBPUSD | Laguerre yeniden giriş | 20 | 2027 | -1,32 | -2,36 |
| USDJPY | Supertrend değişimi | 20 | 1016 | -0,82 | -1,51 |
| AUDUSD | Laguerre yeniden giriş | 20 | 2036 | -1,80 | -3,59 |
| USDCAD | KAMA düzeltmesi | 20 | 1008 | -1,45 | -3,25 |
| USDCHF | Laguerre + KAMA | 20 | 1118 | -1,06 | -2,73 |
| NZDUSD | KAMA düzeltmesi | 20 | 846 | -1,93 | -3,84 |
| EURCHF | Connors dönüşü | 20 | 1342 | -2,10 | -4,24 |

Hiçbir geliştirme ayarı iki alt dönemde birlikte pozitif net ve stres sonucu
vermedi. Seçilenler, başarısızlığı ölçmek için sonraki dönemde yine kontrol
edildi; başarılı strateji diye sunulmadı. Ayrı paritelerdeki işlem sayıları
korelasyon nedeniyle bağımsız örnekler veya ortak portföy getirisi değildir.

## Fiyat dışı alternatif: haber sürprizi

[Andersen ve diğerlerinin çalışması](https://www.nber.org/papers/w8959),
beklenti ile açıklanan değer arasındaki sapmayı kur hareketleriyle ilişkilendirir.
Buradaki para hedefi/stop deneyi o makalenin kârlı işlem sistemi replikasyonu
değildir; araştırmadan esinlenen ayrı bir hipotezdir.

Yerel ForexFactory arşivindeki 10.244 olaydan, önceden tanımlanmış büyüme/işgücü
serilerinde 225 sürpriz açıklama seçildi. Sürpriz büyüklüğü yalnız önceki
açıklamaların hata medyanıyla karşılaştırılır. En az beş dakika beklenir ve
sonraki tamamlanmış 15M sınırından giriş yapılır. Hedef 30 dolar, planlanan
stop 20 dolar, stop 3 ATR, azami süre dört saattir. CPI/faiz/ücretler için
tek bir değişmez yön varsayılmadı; bu seriler dışlandı.

GBPUSD kontrol dönemi (Ekim 2025–11 Eylül 2026):

- 45 varsayımsal işlem; normal masrafla ortalama **+3,02 USD**.
- İki kat masrafla ortalama **+2,11 USD**.
- 30 dolar hedefine ulaşma **%22,2**; bütün pozitif çıkışların oranı **%55,6**.
- Önceki dönemdeki 58 işlem ortalaması **-1,47 USD**.
- Günlük ortalamalar için %95 bootstrap aralığı **[-3,17; +8,54] USD**, p ≈ 0,198.

Bu, her işlemde 30 dolar kazanmak değildir. Az örnek, negatif önceki dönem
ve sıfırı içeren aralık nedeniyle canlı giriş onayı verilmedi. EURUSD/USDCHF'de
de küçük pozitif kontrol ortalamaları vardı; bütün sonuçlar ayrı
`research/news_surprise` raporlarında tutulur. Geçmiş takvim değerlerinin ilk
yayın sürümleri ve gerçek haber anı spread/kaymaları doğrulanmadı.

## İleriye dönük kayıt ve fiyat sınırı

`forex_news_recorder.py` beklentiyi açıklama **öncesinde gerçekten görüldüğü
zamanla**, açıklanan değeri ilk görüldüğü zamanla kaydeder. Önceden kaydedilmiş
beklenti yoksa geçmişe dönük tahmin oluşturmaz. Sonradan revize edilen actual,
ilk görülen değerin veya eski kararın üzerine yazılmaz. Çelişen açıklamalar
henüz gelecekteki bir girişi iptal edebilir; giriş sonrası karar değişmez.
Masraf senaryosu karar sırasında saklanır. Kayıtlar broker emri veya gerçekleşme
değildir; fiyat/lot bilinmediğinde boş bırakılır.

Kaydedici on dakikada bir çalışır. Ayrı Dukascopy fiyat okuyucusu saatte bir
kapanmış mumları kontrol eder; eski fiyat arşivi değiştirilmez. Tam fiyat yolu
sonradan elde edilirse kaydedilmiş gelecekteki niyetin simülasyon sonucu
hesaplanır ve kullanılan mumlar korunur. Eksik gelecekteki fiyatlar uydurulmaz.

**Mevcut sınırlama:** sağlayıcının arşiv, önbelleksiz yakın-tarih ve gerçek-zaman
uçları kontrol edildi. Kapanmış fiyatlarda son kullanılabilir tarih hâlâ
11 Eylül 2026; yakın tarih ve canlı sorgu yeni mum döndürmedi. Fiyat güncelleyicisi
bunu `partial_or_stale_provider_history` diye işaretler. Haber kaydı canlı
çalışıyor; yeni fiyatlar gelmeden ileri sonuç doğrulanamaz.

## Uygulama ve çalıştırma

ML Tahmini → **Yeni indikatörler — internet araştırması ve geçmiş test** alanı
yeni formüllerin kaynaklarını, 168 parite ayarını ve haber sürprizi sonuçlarını
gösterir. Bu sonuçlar mevcut işlem izinlerini açmaz.

```powershell
python -X utf8 research_new_indicators.py
python -X utf8 research_news_surprise.py
python -X utf8 forex_news_recorder.py --watch 600
node tick_downloader/download_forward_prices.js --watch 3600
python -X utf8 -m unittest discover -q
```

Son kontroller: yeni indikatör ve haber sonuçları CSV'lerden yeniden hesaplandı;
formül/gelecek-sızıntısı testleri ve Streamlit panel kontrolü geçti. Toplam
**252 test** başarılı. Bu bütçedeki sonuçlar varsayımsal pip risk ölçeğidir;
gerçek lot adımı, teminat, dönüşüm kuru, spread ve broker gerçekleşmesi ayrıca
değerlendirilmelidir. Gap zararları planlanan stop bütçesini aşabilir.

Ayrıca uyarlanan ortalama için [ASEAN günlük kur çalışması](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0160931)
ve [15M recurrent-learning çalışması](https://link.springer.com/article/10.1007/s40622-026-00479-x)
incelendi. İlki farklı para birimleri/zaman dilimi kullanır; ikincisinde ortalama
test sonucu yalnız iki paritede pozitiftir. Bunlar burada yeniden üretilmiş veya
30 dolar hedefini kanıtlayan sonuçlar olarak sunulmaz.
