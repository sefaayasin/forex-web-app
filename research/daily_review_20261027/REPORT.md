# Bugünkü kazananlar, kaybedenler ve önlenebilir kayıplar

**27 Ekim 2026, saat 17:02:36 Türkiye saati itibarıyla.** Gün tamamlanmadı. MT5 demo hesabının gerçek deal geçmişi ve 382.616 bid/ask ticki incelendi. Bu çalışma sırasında emir gönderilmedi; çalışan botun ayarları veya pozisyonları değiştirilmedi.

## Kapsam ve doğru günlük toplam

Türkiye saatiyle bugün tamamen kapanmış, sanal bakiye ile boyutlandırılan bot pozisyonları seçildi. Önceki gün açılıp bugün kapananlar dahildir. Açık pozisyonlar kazanan/kaybeden sayılmadı. Kaynak [MT5 history_deals_get](https://www.mql5.com/en/docs/python_metatrader5/mt5historydealsget_py); her pozisyonun giriş ve çıkış deal'leri ayrı okunup komisyon, swap ve fee toplandı.

| Ölçüm | Sonuç |
|---|---:|
| Kapanmış işlem | 20 |
| Net kazançla kapanan | 7 |
| Net zararla kapanan | 13 |
| Kazananların net toplamı | +25,13 $ |
| Kaybedenlerin net toplamı | −79,55 $ |
| **Kapanan pozisyonların net toplamı** | **−54,42 $** |
| Pozisyonlara ait komisyon/swap/fee | −21,18 $ |
| Maliyetlerden önce fiyat kâr/zararı | −33,24 $ |

−54,42 $ zaten komisyonları içerir; −21,18 $ tekrar düşülmez. MT5'in yalnız bugün gerçekleşen deal nakit hareketi **−52,35 $**: fark, önceki gün açılan pozisyonların önceki gün ödenmiş giriş komisyonlarının tam pozisyon hesabına dahil edilmesidir. Bu iki toplam farklı kapsamı ölçer.

Önceki gün açılıp bugün kapanan üç işlem **−36,67 $**, bugün açılıp kapanan 17 işlem **−17,75 $** yaptı. Snapshot anında açık boyutlandırılmış bot pozisyonu yoktu.

## Kazandıklarımız

| Kapanış saati | Parite / yön | Net | Kapanış nedeni |
|---|---|---:|---|
| 14:22 | AUDUSD SHORT | +0,14 $ | Takip stopu |
| 15:00 | AUDCAD LONG | +0,33 $ | ML yön dönüşü |
| 15:34 | AUDNZD SHORT | +3,98 $ | Sepet kârı |
| 15:34 | GBPUSD LONG | +8,50 $ | Sepet kârı |
| 17:02 | AUDUSD SHORT | +3,50 $ | Sepet kârı |
| 17:02 | GBPCAD LONG | +7,35 $ | Sepet kârı |
| 17:02 | EURJPY LONG | +1,33 $ | Sepet kârı |

Kazancın **24,66 $'ı sepet çıkışlarından** geldi. Büyük kazananlarda fiyat hedefi aşmış, sepet daha sonra çıkmış. Aynı paritede daha sonra zarar olması, önceki kazanan işlemin giriş koşulunun tekrar işe yarayacağını göstermiyor: GBPUSD aynı gün hem +8,50 $ kazandırdı hem −5,30 $ ve −5,68 $ kaybettirdi.

Kazananların ham ML yön güveni %50,20–52,22 aralığında; ortancası %51,24. Kaybedenlerin ortancası %51,86. Tek günlük örnekte yüksek ham güven kazananları ayırmıyor. Bunlar gerçek kazanma olasılığı olarak kalibre edilmiş skorlar değildir.

## Kaybettiklerimiz: hangi değişiklik neyi önleyebilirdi?

Tablodaki “en iyi net” değeri, gerçek girişten gerçek kapanışa kadar görülen en iyi kapatma tickinin komisyon sonrası yaklaşık değeridir. Sonradan hesaplanan bir üst sınırdır; o tepeyi önceden bilip yakalayan uygulanabilir bir kural değildir. Çapraz paritelerde nakit dönüşüm oranı gerçekleşen işlemden türetildiği için alternatif değerler yaklaşıktır.

| Kapanış | Parite | Net zarar | Kapanmadan önce en iyi net | Değerlendirme |
|---|---|---:|---:|---|
| 00:03 | AUDUSD | −11,84 $ | −1,13 $ | Net kâra hiç geçmedi; daha erken kâr alma kurtaramazdı. Önceki gün açıldı. |
| 00:59 | NZDUSD | −11,97 $ | +2,11 $ | Hedefe ulaşmadan döndü. Girişte maliyet/risk %17,82; bugünkü %15 sınırını bile aşan eski giriş. |
| 04:03 | EURUSD | −12,86 $ | +0,42 $ | %57,64 ham güvene rağmen tam stop. %55 güven filtresi kurtarmazdı. Önceki gün açıldı. |
| 14:40 | GBPUSD | −5,30 $ | +1,20 $ | Küçük kâr sonrası yarım stop; hedef görülmedi. |
| 14:56 | EURUSD | −5,73 $ | −0,56 $ | Net kâra hiç geçmedi; giriş seçimi veya boyutlandırma sorunu. |
| 15:25 | EURCHF | −0,55 $ | +3,37 $ | **Hedef görüldü. Sabit hedefte yaklaşık +2,79 $ olabilirdi.** |
| 16:05 | GBPCAD | −0,07 $ | +4,82 $ | **Hedef görüldü. Sabit hedefte yaklaşık +3,24 $ olabilirdi.** |
| 16:14 | NZDUSD | −6,02 $ | −0,14 $ | Net kâra hiç geçmedi; ham güven %50,03. |
| 16:23 | GBPUSD | −5,68 $ | +0,44 $ | Hedef görülmedi; daha geniş stop önceki tick incelemesinde kaybı büyüttü. |
| 16:27 | NZDUSD | −6,02 $ | −1,26 $ | Net kâra hiç geçmedi; önceki NZD zararından yaklaşık 9 dk 43 sn sonra tekrar LONG. |
| 16:35 | EURCAD | −5,96 $ | +0,08 $ | Komisyon sonrası anlamlı kâr alanı oluşmadı; maliyet/risk %13,18. |
| 16:38 | AUDCAD | −5,68 $ | −1,81 $ | Net kâra hiç geçmedi; maliyet/risk %10,98. |
| 16:44 | AUDJPY | −1,87 $ | +4,09 $ | **Hedef görüldü. Sabit hedefte yaklaşık +2,76 $ olabilirdi.** |

Özetle **13 kaybın 3'ü hedefe ulaştıktan sonra**, **5'i ise hiç net kâra geçmeden** oluştu. Diğer beşinde küçük bir net kâr görülmüş fakat hedefe ulaşılmamış. Bütün kayıpları aynı çıkış ayarıyla açıklamak doğru değil.

## Sabit hedefin kazananlara maliyeti

Yalnız gerçek kapanıştan önce hedefe ulaşan işlemlerde, ilk uygulanabilir bid/ask hedef tickinde çıkış varsayıldı. Diğer işlemlerin gerçek çıkışları aynen tutuldu. Bu, hedefte çıkışın katkısını stop genişletme, ML dönüşü veya sepeti kaldırma gibi ek değişikliklerden ayırır.

- EURCHF, GBPCAD ve AUDJPY'deki üç kayıp pozitife dönerdi: toplam **−2,49 $ yerine yaklaşık +8,79 $**.
- Fakat GBPUSD kazananı **+8,50 $ yerine yaklaşık +3,00 $**, ikinci GBPCAD kazananı **+7,35 $ yerine yaklaşık +2,86 $** olurdu. Diğer kazananlarda da artış/azalış var.
- Bütün aynı girişleri bu şekilde topladığımızda **−54,42 $ yerine yaklaşık −49,81 $**: yaklaşık **4,61 $ iyileşme**, yine zarar.

Bu bir aynı girişler karşılaştırmasıdır. Erken kapanıştan sonra açılabilecek yeni işlemler, para birimi/teminat sınırlarının değişmesi ve farklı sepet kümeleri simüle edilmedi. Bu nedenle toplamı gerçek portföy sonucu olarak kabul etmiyoruz. Fakat “üç zararı sabit hedefle kurtarmak bütün günü kâra geçirirdi” iddiasını desteklemediği açık.

## Girişte kaybedenleri eleyip kazananları koruyabilir miydik?

Aşağıdaki filtreler yalnız giriş anındaki kayıtlı bilgiyi kullanır. Bugün gerçekleşmiş aynı işlemlerin alt kümesidir; yerleri boşalınca hangi yeni işlemlerin açılacağı dahil değildir.

| Filtre | Kalan işlem | Kalan kazanan / kaybeden | Kalan işlemlerin neti | Elenen kazanç / kayıp |
|---|---:|---:|---:|---:|
| Yok | 20 | 7 / 13 | −54,42 $ | — |
| Ham yön güveni ≥ %52 | 8 | 2 / 6 | −26,62 $ | +9,28 $ / −37,08 $ |
| Ham yön güveni ≥ %55 | 1 | 0 / 1 | −12,86 $ | +25,13 $ / −66,69 $ |
| Maliyet/risk ≤ %15 | 19 | 7 / 12 | −42,45 $ | 0 $ / −11,97 $ |
| **Maliyet/risk ≤ %10** | **11** | **6 / 5** | **−8,49 $** | **+3,98 $ / −49,91 $** |

%55 filtresi önceki üç kaybı önlerdi, fakat bugün yedi kazananın tamamını da elerdi. %55'i geçmiş sonuçlara bakarak “kazananları ayıran çözüm” olarak sunmak yanlış. %10 maliyet sınırı bu gün için daha iyi bir ayrım yaptı; %10,10 maliyet oranındaki +3,98 $ AUDNZD kazananı da eleniyor. Bu, %10'u sınırı biraz kaydırarak bugüne uydurmak için gerekçe değildir.

Filtreler birleştirilirse aynı kazananlar korunuyor denemez: %55 ve %10 birlikte bugünkü kapanmış grupta yine yalnız −12,86 $ EURUSD'yi bırakır. Maliyet filtresinin “6 kazanan kaldı” bulgusu, güven filtresi olmadan yapılan ayrı karşılaştırmadır.

## %10 maliyet sınırı geniş geçmişte ne yaptı?

Günlük tablo incelendikten sonra, sonuçlar koşulmadan önce `protocol_cost_cap.json` yazıldı. Canlı kod değiştirilmeden araştırma motoruna isteğe bağlı `max_cost_share` parametresi eklendi. Varsayılan 1,0; önceki deneylerin davranışını değiştirmez. Yeni testte eşleştirilmiş politikalar yalnız maliyet sınırında ayrılır.

28 parite, 1 Ocak 2023–10 Eylül 2026; aynı %65 tarihsel hedef kontrolü, para birimi ve üç pozisyon sınırı, haber filtresi, yarım stop, saatlik ML çıkışı, takas ve sepet. Eski düzende sabit 500 $ boyutlandırma ile %2,5 risk; yeni düzende %1 risk ve %55 ham güven, sabit hedef. Sepet başlangıcı risk bütçesinin 1,2 katı: sırasıyla sermayenin %3 ve %1,2'si. Lot yuvarlama dahil.

| Politika | Maliyet/risk sınırı | İşlem | Yaklaşık haftada | 2023–24 net R/işlem | 2025–26 net R/işlem |
|---|---:|---:|---:|---:|---:|
| Eski takipli, %2,5 risk | %15 | 92.699 | 482 | −0,153 | −0,149 |
| Aynı eski politika | **%10** | 57.631 | 299 | −0,122 | −0,118 |
| Yeni sabit hedef + %55 yön + %1 risk | %15 | 17.300 | 90 | −0,103 | −0,096 |
| Aynı yeni politika | **%10** | 7.708 | 40 | **−0,061** | **−0,064** |

**Daha sıkı maliyet sınırı iki dönemde de işlem başına kaybı azalttı; hiçbir politika pozitif olmadı.** Yeni düzende son dönemde ortalama kayıp yaklaşık üçte bir azaldı. İşlem sıklığı da yarıdan fazla düştü. Sadece daha düşük riskle daha az dolar kaybetmek ile R bazında iyileşmeyi bu nedenle ayrı raporluyoruz.

Bu tarihler önceki araştırmalarda incelendi; yeni bir kör test veya bağımsız kârlılık doğrulaması değildir. Günün sonuçlarından seçilen bir fikri tarih üzerinde tekrar denemenin sınırları [backtest overfitting çalışmasında](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf) açıklanır. Burada istatistiksel anlamlılık veya pozitif getiri iddiası yoktur.

## Mevcut bot açısından anlamı

20 işlemin tamamı eski `trailing` kayıtlarıdır. Son commit ile 16:55'te etkinleşen %55 yön güveni, %1 risk ve sabit hedef düzeniyle açılmış yeni bir işlem bu snapshot'ta yok. 17:02'de kapanan son üç kazanan da eski lot ve çıkış kayıtlarıyla yönetilen pozisyonlardır. Bu günü yeni düzenin canlı başarı/başarısızlık kanıtı sayamayız.

Uygulanabilir araştırma sonucu:

1. Hedefe ulaşıp zararla biten üç işlem için sabit TP mekanik bir çözüm; yeni işlemlerde zaten eklendi. Büyük kazananları küçültmesi de hesaba katılmalı.
2. Girişte maliyet/risk sınırını %15 yerine %10 yapmak, tek günlük örneğin yanında geniş geçmişte de daha az negatif sonuç verdi. Bu, sonraki ileri demo adayıdır; bu çalışmada canlıya uygulanmadı.
3. Net kâra hiç geçmeyen beş işlem için stop/TP ayarı “kârı koruyamaz”. Giriş avantajının gelişmesi veya daha düşük risk gerekir. Daha geniş stopun önceki testleri de bunu çözmedi.
4. Sadece kazananların geçmiş özelliklerini seçerek kalanları da kazanır yapacak doğrulanmış kural bulunmadı. %55 filtresinin bugünkü kazananları da kaldırması buna somut bir örnek.

## Tekrar üretme ve kontrol

```powershell
python -X utf8 research_daily_trade_review.py capture
python -X utf8 research_daily_trade_review.py analyze
python -X utf8 research_daily_trade_review.py history
python -X utf8 -m unittest test_research_bot_backtest test_research_recent_losses
```

`capture` yeni bir snapshot alır; rapordaki 17:02 snapshot'ını tekrar üretmek için mevcut yerel `snapshot.json` ve tick dosyaları korunmalıdır. Ham broker dosyaları git dışında. Ledger'da benzersiz pozisyon ve `net = fiyat kârı + komisyon/swap/fee` eşitliği ayrıca doğrulanır. Araştırma simülatörü ve tick motorunda **14 test geçti**.

Dosyalar: `ledger.csv` tüm kapanmış pozisyonlar; `summary.json` toplamlar; `exit_counterfactuals.csv` sınırlı çıkış denemeleri; `cost_cap_history.json` geniş geçmiş sonuçları. Aktif bot dosyası, emirler, ML modeli veya parametreleri değiştirilmedi; commit oluşturulmadı.
