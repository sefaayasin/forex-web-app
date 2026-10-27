# NZDUSD, GBPUSD, NZDUSD zararlarının MT5 ve geçmiş veri incelemesi

27 Ekim 2026. Üç işlem toplam **−17,72 $** kapandı. Yeni MT5 veri okuması, gerçek bid/ask tickleri ve beş tarihsel çıkış karşılaştırması yapıldı. Çalışan botun ayarları, emirleri veya pozisyonları bu araştırmada değiştirilmedi.

**Bulgu:** İlk iki işlemde stopu genişletmek kaybı büyütürdü. En somut iyileştirme adayı hedefte doğrudan çıkmak; geniş tarihsel testte kaybı azaltıyor, fakat sistemi pozitif beklentiye çevirmiyor. Girişte asgari ML yön güveni kullanmak zayıf işlemleri azaltıyor; önceki geniş test bu yaklaşımın da maliyet sonrası negatif olduğunu gösteriyor.

## Gerçek üç işlem

Pozisyon numarasıyla `history_deals_get` kullanıldı. Açılış ve kapanış komisyonları dahil edildi; ekranın yalnız brüt kâr/zararı kullanılmadı. Saatler Türkiye saatidir.

| İşlem | Giriş | Ham ML yön güveni | Net gerçekleşen | Kapanmadan önce en iyi olası net değer |
|---|---|---:|---:|---:|
| NZDUSD LONG, 411021217 | 15:43:43 | %50,030 | **−6,02 $** | −0,14 $ |
| GBPUSD LONG, 411045182 | 16:05:49 | %52,358 | **−5,68 $** | +0,44 $ |
| NZDUSD LONG, 411059397 | 16:23:54 | %50,408 | **−6,02 $** | −0,98 $ |

Üçünün de bot günlüğündeki kapanış nedeni **YARIM STOP**. İlk ve ikinci NZDUSD arasında yaklaşık 9 dakika 43 saniye var (ilk kapanış 16:14:11). İki NZDUSD işlemi gerçek kapanışlarından önce komisyonu karşılayacak kadar kâra hiç ulaşmamış. Dolayısıyla bu iki girişte “kârı daha erken almak” tek başına çözüm olamaz; girişi atlamak/değiştirmek veya zararı sınırlamak gerekir. GBPUSD'deki +0,44 $ ise sonradan bilinen en iyi tick, yakalanabilir bir kâr garantisi değil.

Girişteki %67–68 **ham ML yön güveni değildir**; sabit hedef/stop araştırmasının geçmiş ulaşma oranıdır. Üstelik gerçek erken çıkış stopun yarısında. Günlükte aynı işlemlerin yarım stopa göre yaklaşık ulaşma değerleri sırasıyla %52,0 / %55,9 / %52,1. Tam stop araştırmasının oranı, bu takipli politikanın güncel kazanma olasılığı olarak sunulmamalıdır.

## Gerçek ticklerle farklı çıkışlar

MT5'ten o gün açılmış ve yakalama anına kadar kapanmış 12 bot işlemi alındı; yalnız üç kayıp seçilmedi. Dokuz paritede **122.372 bid/ask ticki** okundu. Snapshot: **16:38:13 Türkiye saati**. İncelenen alternatiflerde gerçek giriş fill'i, lot ve gerçekleşmiş komisyonlar korundu.

| Alternatif | İlk NZDUSD | GBPUSD | İkinci NZDUSD |
|---|---:|---:|---:|
| Gerçekleşen bot sonucu | −6,02 $ | −5,68 $ | −6,02 $ |
| Her tickte yarım stop kontrolü | −6,02 $ | −5,32 $ | −6,02 $ |
| Stopun %75'ine kadar bekleme | −8,26 $ | −7,57 $ | **−3,78 $ açık değer** |
| Tam stopa kadar bekleme | **−10,50 $** | **−9,82 $** | **−3,78 $ açık değer** |
| Sabit hedef + tam stop | −10,50 $ | −9,82 $ | **−3,78 $ açık değer** |

İkinci NZDUSD'deki alternatifler snapshot anına kadar hedefe veya alternatif stopa ulaşmadı. −3,78 $ varsayımsal açık pozisyon değeridir; kapanmış sonuç veya kurtarılmış işlem değildir. İlk iki işlem ise daha geniş stopu gerçekten görüyor. Yarım stopu kaldırmak bu iki kaybı kurtarmıyor.

Bu replay bir çıkış yolu incelemesidir. Canlı bot beş saniyede bir kontrol eder, replay her ticki görür; GBPUSD'deki 0,36 $ fark bu nedenle ve yürütme farklarıyla tutarlıdır. Sepet çıkışları, saatlik ML dönüşü ve değişen yeni girişler replay'e dahil edilmedi. 12 işlemde replay toplamlarını gerçek portföy performansı diye kullanmamak gerekir; mesela gerçek toplam −22,38 $, yarım stop replay'i −33,04 $. Gerçek sepet kazançlarının replay'de bulunmaması da fark yaratır.

MT5 belgesi tick sorgularında UTC datetime kullanılmasını belirtir. Bu brokerda güncel tick epoch'u gözlenen duvar saatinden **3 saat ilerideydi**; bu nedenle karşılaştırma doğrudan aynı ham `time_msc` eksenindeki deal ve ticklerle yapıldı, yalnız gösterim için gözlenen fark kaydedildi. Genel bir saat düzeltmesi varsayılarak fiyatlar kaydırılmadı. [Resmi tick API belgesi](https://www.mql5.com/en/docs/python_metatrader5/mt5copyticksrange_py).

## Yeni geniş tarihsel çıkış karşılaştırması

İlk dört varyant çalıştırılmadan önce `protocol.json` yazıldı. Ardından, sabit hedefin katkısını yarım stop değişikliğinden ayırmak için **tek ek varyant**, ayrı `protocol_followup_half_target.json` ile, kendi sonucu görülmeden tanımlandı. Bu son varyant ilk dört sonuç incelendikten sonra eklenmiştir.

Mevcut test motoru kullanıldı: 28 parite, 1 Ocak 2023–10 Eylül 2026, ölçülmüş saatlik spread + 0,7 pip komisyon, 5M adım, aynı %65 hedef eşiği, 0,5 ATR takip tabanı, üç pozisyon ve para birimi sınırı, aynı haber filtresi ve sepet takibi. Sabit risk 12,5 $; lot yuvarlama var. R, planlanan tam stop riskine göre net sonuçtur. Buradaki %65 kontrolü deney politikasını temsil eder; başabaş ve pozitif geçmiş kanıtı zorunlu normal mod değildir.

| Kural | İşlem | Yaklaşık haftada | 2023–24 net R/işlem | 2025–26 net R/işlem | Kazanma |
|---|---:|---:|---:|---:|---:|
| Mevcut: %50 erken stop + takip | 96.518 | 502 | −0,158 | −0,152 | %28,9 |
| %75 erken stop + takip | 77.186 | 401 | −0,165 | −0,158 | %33,8 |
| Tam stop + takip | 65.366 | 340 | −0,173 | −0,162 | %36,4 |
| Sabit hedef + tam stop, ML dönüş/takas yok | 77.720 | 404 | **−0,101** | **−0,104** | %68,1 |
| Sabit hedef + mevcut yarım stop, ML dönüş/takas korunur | 137.697 | yaklaşık 716 | −0,111 | −0,109 | %50,2 |

Yarım stopu gevşetmek, kazanma oranını artırsa bile **işlem başına ortalama kaybı büyütüyor**. Sabit hedefte çıkmak işlem başına kaybı azaltıyor; son dönemde −0,152 R yerine yaklaşık −0,104 / −0,109 R. Yarım stoplu sabit hedef daha sık işlem yapıyor; işlem başına iyileşme toplam portföy kaybının aynı oranda azalacağını göstermez. İki sabit hedef varyantı da pozitif değil. Bu yüzden “yarım stopu kaldırdım, artık kaybetmez” veya “hedefte çıkmak kârlı” sonucu çıkarılamaz.

Bu yıllar önceki çok sayıda deneyde incelendi. Dönem ayrımı istikrar kontrolüdür; yeni, dokunulmamış holdout değildir. Sıfırın üstünde yeni bir avantaj gösterilmedi; anlamlılık/güven aralığı iddiası yapılmadı. Aynı tarihlerde tekrar tekrar kural seçme riski, [Backtest Overfitting makalesindeki](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf) gerekçeyle ayrıca dikkate alınmalıdır.

## Giriş filtresi ne kazandırabilir?

Çalışma başında depoda zaten bulunan `research/bot_backtest/results_confidence_fixed_risk.json` yeniden incelendi; bu sonuçlar burada yeniden koşulmuş veya yeni keşifmiş gibi sunulmadı.

| Ham ML yön güveni şartı | Tarihsel yaklaşık haftada işlem | 2025–26 net R/işlem |
|---|---:|---:|
| Ek şart yok | 502 | −0,152 |
| En az %52 | 341 | −0,145 |
| En az %55 | 103 | −0,146 |
| En az %60 | 12 | −0,163 |

**%55 ham yön güveni bu üç zararlı girişi de engellerdi**; hedef ulaşma eşiği %65 bundan ayrı bir parametredir. Ancak bugünkü snapshot'taki 12 kapanmış işlemin tamamı %55'in altında: bu filtre o grupta kazananları da kaldırırdı. Daha geniş testte yaklaşık 103 işlem/hafta kalması, bu adayın her zaman sıfır işlem demek olmadığını gösterir; canlı sıklığı garanti etmez. İşlem başına sonuç ise hâlâ negatif. %55'i “bu üç kaybı önledi” diye doğrulanmış kârlı eşik olarak sunmak doğru değildir.

Zarardan sonra dört saat bekleme de önceki testte işlem başına sonucu kötüleştirmiştir (son dönem yaklaşık −0,155 R). İkinci NZDUSD'yi atlayabilecek olması, bu kuralın genel başarı kanıtı değildir.

## Dolar kaybını azaltmak ile avantaj geliştirmek ayrı şeyler

Mevcut risk bütçesi sanal bakiyenin %2,5'i. Aynı giriş/çıkış fiyatları ve stop mesafeleri korunup sadece yeni işlemin bütçesi %1'e göre lot adımıyla küçültülseydi, bu üç pozisyonun lotları 0,14 / 0,09 / 0,14 yerine **0,05 / 0,03 / 0,05** olurdu. Aynı fiyat yolundaki toplam kayıp yaklaşık **−6,19 $** olurdu (−17,72 $ yerine).

Bu yalnız boyutlandırma karşılaştırmasıdır; yeni hedef seviyeleri, bakiyenin ilerleyişi ve gelecekteki işlem seçimi yeniden simüle edilmedi. Daha az lot dolar kaybını sınırlar, sinyale pozitif beklenti kazandırmaz. Zarar etmeme garantisi yoktur.

## Uygulanabilir sonuç

- **Stopu genişletme adayını ele:** üç gerçek işlem ve geniş tarihsel sonuç bunu desteklemiyor.
- **Giriş kalitesini görünür yap:** yönün ham güvenini, hedefin tarihsel ulaşma oranını ve gerçek çıkış politikasını ayrı raporla. %50,03 ham yön skoru %67 güvenli işlem gibi görünmemeli.
- **Sabit hedef ve %55 ham yön filtresini ancak ileri demo adayları olarak değerlendir:** geçmişte daha düşük işlem kaybı veya daha düşük işlem sayısı sağlıyorlar; hiçbiri bu testlerde pozitif beklenen sonuç üretmiyor. Sonraki deneyde bunlar yeni kurgu olarak dondurulmalı ve gelecek veride ölçülmeli.
- **Amaç aynı girişlerde dolar kaybını azaltmaksa risk boyutu doğrudan kontrol:** %1 bütçe ile aynı üç fiyat yolundaki kayıp örneği yukarıda. Bu araştırmada çalışan botun riski veya açık pozisyonları değiştirilmedi.

## Tekrar üretme ve doğrulama

```powershell
python -X utf8 research_recent_losses.py capture
python -X utf8 research_recent_losses.py replay
python -X utf8 research_recent_losses.py history
python -X utf8 research_recent_losses.py half-target
python -X utf8 -m unittest test_research_recent_losses test_research_bot_backtest
```

`capture` çalışan demo MT5 oturumundan yalnız veri okur; tekrar çalıştırmak snapshot'ı değiştirir. Diğer aşamalar çevrimdışıdır. Araştırma kodunda emir gönderme fonksiyonu yoktur. Ham tick dosyaları ve canlı geçmiş snapshot'ı yerel tutulur; bu klasörün `.gitignore` dosyası onları dışlar.

Çıktılar: `tick_replay.csv`, `focus_findings.json`, `historical_results.json`, `historical_half_target.json`. Testler LONG/BID, SHORT/ASK, çift komisyon, tick sırası, takipte gelecek tepeye bakmama ve ufuk dışı tick kullanmama durumlarını sınar. **13 test geçti.**
