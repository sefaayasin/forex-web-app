# Demo botunun kuralları geçmişte ne yapardı — sonuç

26 Ekim 2026. Sonuç: geçmedi. `forex_bot.py`'nin kuralları 2023-01-01 – 2026-09-10 geçmişinde 500 $ hesabı **ilk iki işlem gününde 250 $'ın, beşinci günde 100 $'ın altına** indiriyor. Yedi varyantın hiçbiri artıya geçmiyor. En iyisi Karar kartının sabit hedefli planı, işlem başına −0,11 R. ML'nin yönüne girmek, tersine girmekten ayırt edilemiyor: −0,187 R'ye karşı −0,192 R.

## Soru

Kullanıcı demo hesapta Karar kuralıyla al-sat yapan bir bot istedi. Ardından çıkış kurallarını kendisi seçti:
- Sabit hedef yok; hedefe gelince stop başabaşa çekiliyor, sonra 1 ATR geriden takip ediyor.
- Stop mesafesinin yarısında çıkış.
- ML yönü dönerse çıkış.
- 4 saatte takılan, zararsız işlemi yeni fırsata bırakma (takas).

Lot 500 $'lık hesaba, %2,5 riske göre ayarlandı. Ayrıca para birimi sınırı, haber ve gün sonu filtresi eklendi. Soru şuydu: bu kurallar geçmişte ne yapardı, hangi çıkış kuralı katkı veriyor?

## Kurgu

Ayrıntı `protocol.json` dosyasında; kurallar sonuçlardan önce yazıldı.

- **Dönem ve kapsam:** 2023-01-01 – 2026-09-10. ML yön modeli 2023 öncesiyle eğitildi; bu dönem modelin görmediği veri. Yön modeli olan 28 parite.
- **Adım:** 5 dakika. Giriş ve açılışta tetiklenen çıkışlar 5M açılışından.
- **Giriş:** Botla aynı hesap:
  - Stop 1,5 × 15M ATR14, lot riskten.
  - Hedef 0,3 R.
  - Hedefin stoptan önce gelme oranı ≥ %60 ve maliyet hedefin yarısından az.
- **Filtreler:** En fazla 3 pozisyon. Para biriminde aynı yönde en fazla 1. Teminat (1:200) bakiyenin yarısını geçmez. Parite başına saatlik sinyalde tek giriş. New York gün sonunda ve hafta sonu kapanışında girilmez. Paritenin para birimlerinde yüksek etkili haberden (ForexFactory arşivi) 60 dk önce ile 15 dk sonrası arasında girilmez.
- **Maliyet:** Girişin New York saatindeki ölçülmüş spread + 0,7 pip komisyon.
- **Mum içi sıra:** Aynı mumda hem kötü hem iyi taraf görüldüyse önce kötü olan sayıldı.
- **İki görünüm:**
  - **Bileşik:** 500 $'dan başlar, %2,5 risk, sonuçlar bakiyeye eklenir. Protokoldeki ana görünüm.
  - **Sabit risk:** Her işlemde 12,5 $, bileşik büyüme yok. Hesap erken eridiği için tüm dönemi ölçmek amacıyla sonradan eklendi; kurallar aynı.

## Sonuçlar

**Sabit risk, tüm dönem** (işlem başına ortalama, risk biriminde):

| Varyant | İşlem | Günde | Ortanca süre | Kazanma | Ortalama |
|---|---:|---:|---:|---:|---:|
| Bot (şu anki kurallar) | 121.569 | 126 | 15 dk | %32 | **−0,187 R** |
| Sabit hedef (Karar kartı planı) | 81.452 | 85 | 20 dk | %68 | **−0,109 R** |
| Bot, yarım stop yok | 77.722 | 81 | 25 dk | %41 | −0,199 R |
| Bot, ML dönüş çıkışı yok | 113.585 | 118 | 15 dk | %32 | −0,189 R |
| Bot, takas yok | 121.452 | 126 | 15 dk | %32 | −0,187 R |
| Bot, ML'nin ters yönü (kontrol) | 124.502 | 129 | 15 dk | %29 | −0,192 R |
| Bot, +0,5 pip kayma (stres) | 112.526 | 117 | 15 dk | %38 | −0,199 R |

Her yıl ve her varyantta ortalama aynı aralıkta. Bot için yıllar: 2023 −0,189 · 2024 −0,191 · 2025 −0,182 · 2026 −0,188.

**Bileşik, 500 $ başlangıç:** Yedi varyantın hepsi 2023 içinde 12–18 $'a iniyor (−%96 / −%97). Bot varyantında bakiye 3 Ocak 2023'te 250 $'ın, 5 Ocak'ta 100 $'ın altında. Sonrasında lot 0,01'in altına düştüğü için neredeyse işlem açılamıyor.

**Bot, sabit risk, kapanış sebebine göre** (girişle aynı mumda kapananlar hariç):

| Sebep | Ortalama |
|---|---:|
| Takip stopu (hedefe geldi, sonra takip) | +0,195 R |
| Yarım stop | −0,563 R |
| ML yönü döndü | −0,069 R |
| 24 saat doldu | −0,452 R |
| Takas (daha iyi fırsat) | +0,097 R |

## Yorum

- **Takip, kazancı sıfıra yakın bırakıyor.** Hedef girişten yaklaşık 0,45 ATR uzakta, takip mesafesi ise 1 ATR. Fiyat hedefe gelince stop başabaşa çekiliyor. Stopun başabaşın üstüne çıkması için fiyatın hedefin iki katından fazla gitmesi gerekiyor. Takiple kapanan işlemlerin ortancası 0 R.
- **Yarım stop, ulaşma oranını düşürüyor.** Hedef (0,45 ATR) ile yarım stop (0,75 ATR) yakın; işlemlerin yarısı yarım stopta, yarısı başabaş civarında bitiyor. Yapı, kazananlar ≈ 0 R ve kaybedenler −0,56 R. Maliyetten önce bile kazanamaz.
- **İşlem sıklığı maliyeti katlıyor.** İşlemler dakikalar içinde kapanıyor, boşalan yer bir sonraki 5M'de hemen doluyor. Sonuç: günde 80–130 işlem ve her birinde spread + komisyon.
- **Yön bilgisi yok denecek kadar az.** ML'nin yönü ile tersi arasındaki fark işlem başına 0,005 R. Bu, daha önceki ölçümle tutarlı: ML 4 saatte yazı-turadan 2–3 puan iyi, maliyetin yanında çok küçük. Çıkış kuralları bu yokluğu kâra çeviremez.
- **En az kötü varyant sabit hedef (−0,11 R).** %68 kazanma, ama kayıplar kazancın üç katı. Başabaş için %76 gerekiyor.
- **Karamsar mum içi sıra sonucu değiştirmiyor.** Bot işlemlerinin %20'si girişle aynı mumda kapanıyor; bunlar çıkarılınca ortalama −0,173 R. Sabit hedefte aynı mumda kapananlar çoğunlukla hedef işlemleri; onlar çıkarılınca sonuç −0,148 R'ye kötüleşiyor.

## Uygulamaya

- **Demo bot:** Protokoldeki karar kuralı uygulandı ve sonuç demo sonuçlarından önce kullanıcıya bildirildi. Bu kurallarla çalışan demo bot, hesabı geçmişteki gibi eritecektir.
- **Kural değişikliği yok:** Sonuçlara bakılarak kural ya da parametre değiştirilmedi. Değiştirilecekse yeni ve önceden yazılmış bir testle sınanacak.

## Sınırlar

- Anlık spread sıçraması, kayma (stres varyantı hariç) ve broker reddi modellenmedi. Bot dakikada bir bakıyor; test 5 dakikada bir.
- Giriş eşiği için kullanılan `trade_plan_stats` aynı dönemden ölçüldü. 0,3 R hedefte oranlar neredeyse her zaman eşiğin üstünde olduğundan etkisi küçük.
- Pip değeri ve teminat, USD paritelerinin 5M açılışlarından hesaplandı.
- Bileşik görünümde düşüş, günlük gerçekleşmiş bakiyeden hesaplandı; açık pozisyonların anlık zararı dahil değil.

`python -X utf8 research_bot_backtest.py` (bileşik) ve `--fixed-risk` (sabit risk). Sonuçlar `results.json` ve `results_fixed_risk.json` dosyalarında.

## Ek: "3 işlemden toplam 15 $ olunca çık" (sepet kuralı)

26 Ekim 2026. Kullanıcının fikri: açık pozisyonların toplam net kârı +15 $'a (500 $'ın %3'ü) ulaşınca hepsini kapat, yeni işlemlere geç. Kurallar sonuçlardan önce `protocol_basket.json`'a yazıldı.

**Fiyat sınırları.** 5M mumlarda toplam kârın mum içinde ne zaman 15 $'a dokunduğu bilinemiyor. Bu yüzden iki sınır hesaplandı:
- **Temkinli:** Toplam, her mum açılışında kontrol edilir.
- **İyimser:** Mum içinde her pozisyonun en iyi fiyatı aynı anda görülmüş sayılır.

**Sabit risk görünümü** (12,5 $ risk, 15 $ toplam hedef, 2023–2026, işlem başına ortalama):

| Varyant | Temkinli | İyimser |
|---|---:|---:|
| B1 toplam +15 $ + botun kuralları | −0,185 R | −0,133 R |
| B2 toplam +15 $ + yalnız tam stop | **−0,095 R** | +0,050 R |
| B3 toplam +15 $ + yarım stop | −0,108 R | −0,005 R |
| B4 B2, ML'nin **ters** yönü (kontrol) | −0,120 R | **+0,036 R** |

**B2, temkinli:**
- Haftada 213 işlem.
- Toplam hedefle kapanan pozisyon ortalaması +7,40 $ (22.429 pozisyon); stop ortalaması −11,95 $ (18.005 pozisyon).
- 3,7 yılda −46.518 $, yani işlem günü başına yaklaşık −48 $.

**Bileşik görünüm (500 $):** Temkinli varyantların hepsi hesabı 12–14 $'a indiriyor.

**Karar.** Protokoldeki karar kuralı yalnızca iyimser sınıra bakıyordu ve B2'nin iyimser sınırı artıda. Ama aynı iyimser varsayım, **ters yöne giren kontrolü de (B4) +0,036 R kârlı gösteriyor**. Yanlış yöne girmenin kârlı görünmesi, bu kârın stratejiden değil varsayımdan geldiğini gösterir: üç pozisyonun mum içindeki en iyi anları gerçekte aynı anda olmaz ve bu "en iyi" fiyatlardan çıkılamaz. ML yönünün katkısı her iki sınırda da yaklaşık +0,015–0,025 R. Yönü olmayan bir girişin gerçek beklentisi ise maliyet kadar eksidir. Bu yüzden B2'nin gerçekçi beklentisi yaklaşık −0,03 ile −0,05 R arasında.

Sonuç: Sepet kuralı **kazandırmıyor**. Zararı botun şu anki kurallarına göre azaltıyor; işlem başına −0,19 R yerine −0,1 R civarı. Ama artıya geçirmiyor. Bu karar, protokoldeki kuralın kontrol varyantıyla birlikte yorumlanmasıdır ve burada açıkça not edilmiştir. Eşik ve varyantlar değiştirilmedi.

`python -X utf8 research_bot_backtest.py --basket --fixed-risk` ve `--basket`. Sonuçlar `results_basket_fixed_risk.json` ve `results_basket.json` dosyalarında.

## Ek 2: "Toplam 15 $ civarında dolanıp düşecek gibiyse çık" (sepet takibi)

27 Ekim 2026. Canlı bota eklenen `BasketWatch` kuralı test edildi. Kural şöyle çalışıyor:
- Açık pozisyonların komisyon dahil toplam kârı bakiyenin arm %'sini gördükten sonra izlenir.
- Tepeden "geri" kadar geri verirse hepsi kapanır.
- 10 dk yeni tepe yapmayıp son 5 dk aşağı dönerse de hepsi kapanır.

Kurallar sonuçlardan önce `protocol_basket_trail.json`'a yazıldı. Kontroller 5M açılışlarında yapıldı; canlı bot 5 sn'de bir bakıyor. Ayar 2023–2024 verisiyle seçildi, 2025–2026 verisiyle doğrulandı. Her ayar ML'nin ters yönüyle de çalıştırıldı.

**Sabit risk görünümü** (12,5 $ risk, işlem başına ortalama):

| Taban · ayar | 2023–24 | 2025–26 | Ters yön 2025–26 |
|---|---:|---:|---:|
| Bot kuralları · arm %3, geri %30 (**canlı ayar**) | −0,187 R | −0,183 R | −0,190 R |
| Bot kuralları · arm %2, geri %20 (2023–24'te en iyi) | −0,186 R | −0,182 R | −0,187 R |
| Yalnız tam stop · arm %5, geri %20 (2023–24'te en iyi) | −0,089 R | −0,102 R | −0,133 R |
| Karşılaştırma: bot, sepet yok (Ek'ten önceki tablo) | | −0,187 R (tüm dönem) | |

Izgaradaki 18 ayarın hiçbiri, hiçbir dönemde artıya geçmiyor. Bot kuralları tabanında 9 ayar −0,184 ile −0,186 R arasında; yalnız tam stop tabanında −0,095 ile −0,105 R arasında.

**Neden etkisiz.** Canlı ayarda sepet kuralı 3,7 yılda 3.986 pozisyon kapattı, pozisyon başına ortalama +5,90 $. Ama aynı dönemde 58.195 pozisyon yarım stopta −6,72 $ ile kapandı. Botun kendi çıkışları pozisyonları dakikalar içinde kapattığı için toplam nadiren 15 $'a geliyor. Geldiğinde kârı korumak, kayıp tarafını değiştirmiyor.

**Karar.** Protokoldeki şart (2025–2026'da > 0 R ve ters yönden yüksek) hiçbir tabanda sağlanmadı. Canlı botun kuralları ve ayarları değiştirilmedi. Sepet takibi canlı botta kalıyor: bota zarar vermiyor (−0,185 R'ye karşı −0,187 R), kazandırmıyor da.

`python -X utf8 research_bot_backtest.py --basket-trail --fixed-risk`. Sonuçlar `results_basket_trail_fixed_risk.json` dosyasında.

## Ek 3: ML girişine 15M/5M skoru ve oynaklık desteği

27 Ekim 2026. Kullanıcı, ML yön girişine üç destek eklenmesini istedi:
- Uygulamanın 15M teknik skoru, giriş yönünde ≥ 25.
- Aynı skor 5M'de.
- 4 saatlik oynaklık modeli: yüksek (olasılık ≥ 0,5) ya da düşük.

Kurallar sonuçlardan önce `protocol_support.json`'a yazıldı; ayar seçimi yapılmadı. Taban, canlı botun kuralları artı sepet takibi. Görünüm sabit risk (12,5 $). Her varyant ML'nin ters yönüyle de çalıştırıldı. Ters yönde destek şartları giriş yönüne göre uygulandı.

| Varyant | Haftada | Kazanma | 2023–24 | 2025–26 | Ters yön 2025–26 |
|---|---:|---:|---:|---:|---:|
| S0 destek yok | 658 | %31,9 | −0,187 R | −0,183 R | −0,190 R |
| S1 ML + 15M | 482 | %30,2 | −0,207 R | −0,208 R | −0,204 R |
| S2 ML + 15M + 5M | 434 | %29,7 | −0,213 R | −0,210 R | −0,210 R |
| S3 ML + oynaklık yüksek | 218 | %29,7 | −0,200 R | −0,205 R | −0,196 R |
| S4 ML + oynaklık düşük | 621 | %32,0 | −0,192 R | −0,187 R | −0,193 R |
| S5 ML + 15M + 5M + oynaklık yüksek | 96 | %28,4 | −0,208 R | −0,214 R | −0,205 R |
| S6 ML + 15M + 5M + oynaklık düşük | 383 | %30,0 | −0,214 R | −0,210 R | −0,211 R |

**Karar.** Hiçbir varyant iki dönemin hiçbirinde artıya geçmedi; canlı botun giriş kuralı değiştirilmedi.

**Gözlemler.**
- **15M/5M desteği botu kötüleştiriyor:** −0,185 R'den −0,21 R'ye. Skor giriş yönünde güçlüyken fiyat çoğunlukla hareketin sonundadır. Bu, tazelik araştırmasıyla (research/freshness) tutarlı.
- **ML'nin katkısı kayboluyor:** Destekli varyantlarda ML'nin yönü ile tersi aynı sonucu veriyor.
- **Oynaklık:** Yüksek oynaklıkta girmek kötüleştiriyor (−0,20 R). Düşük oynaklıkta girmek desteksiz girişle aynı.

`python -X utf8 research_bot_backtest.py --support --fixed-risk`. Sonuçlar `results_support_fixed_risk.json` dosyasında.
