# 1000 dolar hesapta 30 dolar net hedef için giriş araştırması

8 Ekim 2026. Bütçe: 1000 dolar hesap; işlemde en fazla 100 dolar planlanan stop
zararı; 30 dolar net hedef. **Denenen yeni yöntemlerden güvenilir ve maliyet
sonrası kârlı bir LONG/SHORT sistemi çıkmadı.** Hesaplama sorunları ve negatif
geçmiş sonuçla giriş öneren işlem planı düzeltildi.

## Teknik denetim ve düzeltmeler

- EMA eğimi canlıda 7/19, backtestte 8/20 mumdan hesaplanıyordu. MACD geçiş
  sayımı da ilk geçişi farklı sayabiliyordu. Aynı EURUSD verisinde 100 kontrolden
  ikisi farklıydı (-35/-45 ve -65/-75). Skor artık `bias_score_frame` üzerinden
  ortak hesaplanır. Aynı 100 kapanışta sıfır farkla doğrulandı.
- RSI/ATR'nin Wilder ortalaması ilk gözlemle başlıyordu. İlk tam pencerenin
  aritmetik ortalamasıyla başlatıldı; sonra Wilder tekrarı uygulanıyor. RSI,
  ATR, EMA/MACD ve Bollinger bağımsız skaler formüllerle kontrol edildi.
  İlk ATR true-range değeri ilk mumun yüksek-düşük aralığıdır; sabit fiyat
  RSI'si bilinçli olarak nötr 50 kabul edilir.
- 365 günlük giriş testinde 1H/4H yalnız 90/120 günden alınıyordu. Önceki
  dönemde filtre verisi bulunmadığından girişler engelleniyordu. Artık istenen
  süre + ısınma geçmişi alınır: 1H 385, 4H 425 gün. [Yeniden hesaplanan gerçek
  motor testi](../adaptive_entry/REPORT.md) 109 işlem üretti; parite başına
  yeterli örnek yok ve yalnız EURUSD pozitiftir.
- Negatif geçmiş ortalamayla bile işlem planında LONG/SHORT eylemi çıkabiliyordu.
  Negatif, doğrudan test edilmemiş veya yalnız betimsel geçmiş sonucu olan plan
  giriş önermez. Yakın hedef/stop oranı artık doğrudan ölçülmüş oran sayılmaz.
  Lot 0,01 adımında aşağı yuvarlanır; planlanan nakit risk bütçesini aşmaz.
  Bütçe minimum lota yetmiyorsa plan açılmaz. Eski backtest kalite önbelleğinin
  yeni hesapla eşleşmemesi için sonuç anahtarı sürümlendi.

Doğru formül, doğru piyasa tahmini anlamına gelmez. Teknik skor geçmiş fiyat
durumunun özetidir; 80/100 puan %80 kazanma ihtimali değildir.

## Doğrudan dolar hedefiyle test

Yedi ana parite; 15M girişler; 5M işlem yolu; 2008–11 Eylül 2026 yerel arşivi.
13 giriş kuralı: mevcut skor, skorun tersi, EMA düzeltmesi, trend içinde RSI2,
RSI2 dönüşü, RSI14 yeniden giriş, Bollinger yeniden giriş, Bollinger + ADX yatay
rejim, Stochastic, Donchian kırılımı, ters Donchian, EMA20/50 kesişimi ve Keltner.
Stop: 1,5 / 3 / 6 ATR; azami süre: 1 / 4 saat. Parite başına 78,
toplam **546 ayar** denendi.

Giriş, kapanmış 15M mumundan sonraki 5M açılışı; UTC 07:00–17:00 seansı;
parite başına tek pozisyon. Eksik/tamamlanmamış işlem yolları ve bozuk fiyat
pencereleri dışlanır. 2016–2018 ve 2019–2022 alt dönemlerinin her birinde en az
100 işlem gerekir. İki kat masrafla en kötü alt dönem ortalaması en yüksek olan
aday seçilir. 2023 sonrası sonuçlar seçime girmez.

Masraf: girişin New York saatindeki ölçülen spread + 0,7 pip komisyon; stres:
iki katı. 2026 masraf profili eski fiyatlara uygulanan açık bir varsayımdır;
tarihsel gerçek broker masrafı olduğu iddia edilmez. Stop maliyet dahil
100 dolar, hedef maliyet sonrası 30 dolar ölçeğine çevrilir. Stop 10 pip ve
masraf 2 pip ise brüt hedef 5,6 piptir. Aynı mumda iki seviyeye temas: stop
önce. Ters gap açılış fiyatından gerçekleşir.

| Parite | Geliştirmede seçilen kural | Kontrol işlemi | 30 $ hedefe ulaştı | Ortalama net $ | İki kat masrafla $ |
|---|---|---:|---:|---:|---:|
| EURUSD | Trend içinde RSI2 | 720 | %20,1 | -1,31 | -3,95 |
| GBPUSD | EMA20/50 kesişimi | 739 | %45,5 | -5,72 | -8,31 |
| USDJPY | Donchian kırılımı | 2096 | %46,6 | -1,88 | -3,65 |
| AUDUSD | EMA20/50 kesişimi | 644 | %41,1 | -5,77 | -10,38 |
| USDCAD | Bollinger + yatay rejim | 218 | %49,1 | -6,36 | -11,14 |
| USDCHF | Bollinger + yatay rejim | 173 | %50,3 | -5,45 | -9,88 |
| NZDUSD | Donchian kırılımı | 1913 | %38,2 | -7,21 | -12,25 |

Toplam **6503 kontrol işlemi**; pariteler korelasyonlu, bağımsız örnekler değil.
Yedi adayın yedisi de maliyet sonrası negatif. Hiçbiri bütün başarı ölçütlerini
geçmedi: iki geliştirme döneminde pozitif net/stres sonuç; en az 300 kontrol
işlemi; pozitif günlük blok-bootstrap alt sınırı; yedi parite için düzeltilmiş
p ≤ 0,05; her kontrol yılında pozitif sonuç. Geliştirmede de negatif adaylar
başarısızlığı görünür kılmak için yine kontrol edildi; başarılı diye sunulmadı.

Yalnız hedef/stop çıkışlarında başabaş isabet:
**100 / (100 + 30) = %76,9**. Zaman aşımı çıkışlarının tutarı değişir ve ayrıca
hesaplanır. Hedefe ulaşma oranı, bütün kârlı çıkışların oranı veya fiyat yönü
olasılığıyla karıştırılmaz.

1000 dolar hesap, sabit 100 dolar riskle sonuçları sırayla uygularsa her seçilmiş
yöntemde bir noktada bakiye yeni 100 dolar risk için uygunsuz olur. Bu noktadan
sonra hesap simülasyonu yeni işlem açmaz. Tablo bütün varsayımsal fırsatların
ortalamasıdır; iflas sonrası gerçek hesap işlemleri veya portföy getirisi değildir.
Daha düşük risk bütçeleri bu 546-ayarlı aramada ayrıca seçilmedi; sonuçlar
30/100 senaryosunadır.

## Hedefe göre kendi kendine öğrenme

`forex_adaptive.py --net-target-ratio 0.3` eklendi. Başarı etiketi **hedefe stop
veya süre sonundan önce ulaşmak**; süre sonunda küçük pozitif sonuç almak hedef
başarısı değildir. Adaylar 15M skorunun ±25 geçişleridir; yukarıdaki 13-kural
testinden farklı takvimdir. Stop 1,5 ATR, en fazla dört saat; net hedef/risk 0,3.

Üç/sekiz yıl geçmiş, fiyat/teknik/geniş indikatör grupları, lojistik/ağaç modeli,
zaman sıralı seçim ve kalibrasyonla her üç ayda yeniden öğrenilir. Eşik yalnız
geçmiş net/stres kârından seçilir. Yeterli eşik yoksa giriş seçilmez. Masraf
senaryosu burada sabit 1,5 pip; stres 3 pip. Sonuçlar `research/adaptive_cash_entry`
altında. ML ekranı bu hedefe uygun modelin raporunu ve katsayılarını gösterir.

Son hedef-odaklı öğrenmede 2023+ için **35.903 aday ve 840 model karşılaştırması**
kontrol edildi. Geliştirmede net/stres kâr koşullarını sağlayan bir eşik çıkmadı;
yedi paritenin tamamında son model giriş seçimini kapalı tuttu. Zorunlu seçim
yapan teşhis filtreleri de yedi paritenin tamamında negatifti. Brier kaybı
basit eğitim oranı referansına göre azaldı, fakat bu küçük iyileşme pozitif
işlem beklentisi oluşturmadı.

```powershell
python -X utf8 research_cash_target.py
python -X utf8 forex_adaptive.py --net-target-ratio 0.3 --output research/adaptive_cash_entry
python -X utf8 forex_adaptive.py --net-target-ratio 0.3 --output research/adaptive_cash_entry --watch 600
python -X utf8 audit_current_entry_engine.py
python -X utf8 -m unittest discover -q
```

İzleme, bilgisayar ve süreç açıkken yerel dosya değişikliklerini on dakikada
bir kontrol eder. Yeni veriyle yeniden öğrenir; fiyat indirmez ve broker işlemi
başlatmaz. Yeni fiyat gelmezse yeni piyasa deneyimi öğrenilmiş sayılmaz.

Özet → İşlem Planı varsayılanı 30 dolar net hedef / 100 dolar planlanan stop.
Hedefinin geçmiş testi alanı ilgili paritenin araştırma sonucunu gösterir.
Doğrulanmamış planlar BEKLE gösterir; teknik yön bilgi olarak kalır.

## Doğrulama ve sınırlar

**234 test geçti.** Bağımsız indikatör formülleri, canlı/backtest uyumu, yeterli
filtre geçmişi, geleceğe bakmama, masraf dahil hedef, küçük zaman aşımı kârını
hedef başarısı saymama, aynı-mum stop önceliği, gap, bütçe yuvarlama ve
doğrulanmamış girişleri engelleme sınandı. Yedi cash kaydı CSV'lerden yeniden
hesaplandı. İşlem planı ve kanıt paneli Streamlit AppTest ile hatasız açıldı.

- 2023+ tarih daha önce incelendi; yeni deney bunu kör/ileriye dönük test yapmaz.
  Oluşurken kaydedilecek demo sinyalleri, gerçek broker verisi ve masrafları
  ayrıca doğrulanmalıdır.
- Dolarlar sabit pip risk ölçeğidir; gerçek lot adımı, kaldıraç/teminat ve USD
  dışındaki karşı para biriminin çıkıştaki dönüşümü araştırmada yoktur.
  Tablo gerçek broker P&L'si değildir.
- Gap, kayma ve değişen masraf planlanan 100 dolar zararı aşabilir. Stop fiyatı
  kesin dolar zarar garantisi değildir.
- Her LONG/SHORT çağrısında en az 30 dolar kazanma garantisi bulunmaz. Bu
  çalışma başarısız denemeleri kârlı sistem gibi raporlamaz.

Formül referansı: [TA-Lib RSI / Wilder tanımı](https://ta-lib.org/functions/rsi.html).
Zaman sıralı ayrım: [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
Olasılık kalibrasyonu: [scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html).
