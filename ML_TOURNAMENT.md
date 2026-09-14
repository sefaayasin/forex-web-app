# Otomatik ML karşılaştırma laboratuvarı

## Tamamlanan deneyin sonucu — 14 Eylül 2026

1.680 geliştirme eğitimi, 28 güven eşiği seçimi eğitimi ve 56 final eğitimi
tamamlandı. Yeterli geçmişi olan 28 parite kontrol edildi; EURZAR dışlandı.
Paritelere eşit ağırlık verilen 2023+ kontrol sonuçları:

| Kullanım | Model | Accuracy | Dengeli accuracy | Kapsam |
|---|---|---:|---:|---:|
| 4 mum sonra fiyat yönü | Extra Trees + bağlam/FOMC | %52,79 | %52,40 | %100 |
| 72 mum sonra yüksek oynaklık | Lojistik regresyon + bağlam/FOMC | %67,70 | %62,49 | %100 |
| Aynı oynaklık modeli, güven eşiği 0,65 | Aynı model | **%79,23** | %68,90 | **%45,71** |

Son satır tüm veri üzerinde %79 doğruluk anlamına gelmez. Yalnız geliştirmede
belirlenen güven filtresinin tuttuğu örnekleri kapsar. Aynı seçilmiş örneklerde
süreklilik referansı %71,74, eğitim çoğunluğu referansı %71,06 doğruluk verdi.
Tam örneklem oynaklık referansları sırasıyla %62,02 ve %64,18'dir.
Yön için 73.362, oynaklık için 7.681 örtüşmeyen sonuç penceresi ölçüldü;
pariteler arasındaki korelasyon nedeniyle bunlar tamamen bağımsız örnekler değildir.

Yön modelinin 0,50 eşiğinde işlem başına ortalama sonucu maliyetsiz +0,69 baz puan,
1,5 pip maliyette **-0,77 baz puan**, 3 pip maliyette -2,23 baz puandır.
Dolayısıyla bu araştırma doğrudan işlem açan bir yön modeli için yeterli ekonomik
kanıt üretmedi. En anlamlı aday **oynaklık/risk desteği** oldu.

Özellik heatmap'inde en büyük katkı geçmiş oynaklık ve çoklu pencere bağlamından
geldi: 72 mumluk oynaklık hedefinde fiyat grubuna göre ortalama +8,95 dengeli
doğruluk puanı; FOMC de eklenince toplam +9,96 puan. Bu rakamlar geliştirme
dönemlerindeki tüm model ailelerinin ortalamasıdır. Haber tek başına kazanımı
açıklamıyor; COT eklemek tutarlı ilave fayda göstermedi.

39 birim/regresyon testi, Streamlit paneli ve heatmap seçimi geçti.
56 final ölçüm satırının accuracy ve dengeli accuracy değerleri kaydedilmiş
tahminlerden bağımsız olarak yeniden hesaplanıp doğrulandı.

## Çalıştırma ve çıktılar

```powershell
pip install -r requirements-ml-research.txt
python forex_ml_tournament.py
python -m unittest -v test_forex_ml_tournament.py test_forex_ml.py test_forex_decision_core.py test_forex_modules.py
```

Ana rapor: `data/ml/tournament/report.html`. İnternet bağlantısı olmadan açılır;
grafikler etkileşimlidir. `figures/` altında paylaşılabilir PNG heatmap'leri vardır.
Streamlit uygulamasındaki “ML karşılaştırma laboratuvarı” bu sonuçları okur.
Ana uygulamanın çalışması CatBoost veya XGBoost kurulmasını gerektirmez;
bu bağımlılıklar yalnız çevrimdışı eğitim içindir.

Tamamlanan çalışma tekrar çalıştırılırsa son test yeniden eğitilmez; kayıtlı
sonuçlardan görseller üretilir. Kesilen geliştirme koşuları JSONL kaydından sürer.
Kaynak veri hash'leri veya protokol değişirse yeni `--output` klasörü gerekir.
Yeni klasör açmak daha önce incelenmiş final verisini yeniden kör test yapmaz.

## Arama kapsamı

- Yedi ana parite: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD.
- Beş aile: ölçeklenmiş lojistik regresyon, histogram gradient boosting,
  Extra Trees, CatBoost, XGBoost. Parametreler kaynakta sabittir.
- Dört grup: fiyat/teknik, geçmiş 4–480 mum bağlamı, bağlam + FOMC,
  bağlam + FOMC + gecikmeli COT duyarlılık deneyi.
- Üç ufuk: 4, 24, 72 saatlik mum.
- İki hedef: sonraki açılıştan ufuk sonuna fiyat yönü;
  gelecek ufuktaki gerçekleşen oynaklığın son 480 mumluk seviyeyi aşması.
- İki genişleyen geliştirme testi: 2016–2018 ve 2019–2022.

Toplam 1.680 geliştirme eğitimi. Seçilen modeller için güven eşiği ayrıca aynı
geliştirme dönemlerinde belirlenir. Sonrasında her hedef için tek konfigürasyon
tüm yeterli geçmişe sahip paritelere uygulanır. Final sonuçları görüldükten sonra
pariteye göre model, özellik veya eşik değiştirilmez.

Bu arama tüm olası algoritmaların matematiksel optimumunu kanıtlamaz; belirtilen
adaylar arasında tekrar üretilebilir bir karşılaştırmadır. 5m verisinin milyarlarca
penceresini denemek daha iyi doğrulama anlamına gelmez; ilk kapsam saatlik veridir.

## Model ve eşik seçimi

Ana ölçüt, parite/dönemlere eşit ağırlıklı ortalama **balanced accuracy** eksi
bu skorun standart sapmasının yarısıdır. Böylece yalnız bir dönemde çok iyi olan
adayın seçilmesi zorlaşır. COT'un gerçek yayın zamanları doğrulanmadığından
COT deneyleri final model seçimine alınmaz.

Seçilen modelin güven eşiği 0,50–0,90 arasında geliştirmede karşılaştırılır.
Her parite/dönemde en az %20 kapsam, en az 30 örnek ve iki gerçek sınıf bulunmalıdır.
Eşik aynı dengeli doğruluk/istikrar ölçütüyle belirlenir; seçilmiş örneklerin
accuracy'si tam örneklem accuracy'sinin yerine geçmez. `abstention_development.csv`
bu karşılaştırmanın tamamını, `selected.json` dondurulmuş seçimi içerir.

## Sızıntı ve veri kalitesi kontrolleri

- Yalnız tamamlanan saatlik mumdan özellik çıkarılır; giriş sonraki açılıştır.
- Etiket bitiş zamanı bölüm sınırından önce olmalıdır; boşluk sadece rastgele
  satır sayısı olarak uygulanmaz. Rastgele train/test bölmesi yapılmaz.
- Ölçekleme ve eksik değer doldurma yalnız eğitimde öğrenilir.
- Üst zaman ölçekleri geçmiş saatlik pencerelerden oluşturulur; henüz kapanmamış
  günlük/haftalık mumun gelecekteki değerine bakılmaz.
- FOMC gün bilgisi ertesi UTC gece yarısından itibaren kullanılabilir varsayılır.
- COT yalnız 14 günlük varsayımsal gecikme ile ayrı duyarlılık deneyidir.
- FRED gözlem tarihleri, revize makro değerler ve haber sonrası etki sütunları
  gerçek zamanlı haber özellikleri olarak kullanılmaz.
- Kaynak CSV'ler değişmez. Sıralama bellekte yapılır; yinelenen tarihler reddedilir.
- Bir fiyat adımına kadar OHLC sınır sapması bellekte düzeltilip sayılır.
  Daha büyük sınır hatası içeren mumlar, onlardan geçen etiketler ve sonraki
  1.000 özellik ısınma mumu karantinaya alınır. Sıfır/negatif/sonsuz fiyat reddedilir.
- Ufuk + 96 saatten uzun sonuç pencereleri eksik geçmiş olarak dışlanır.

Final accuracy ve güven aralıkları örtüşmeyen sonuç pencereleri üzerinde hesaplanır.
Geliştirme modelleri sabit UTC 4 saatlik örnekleme takvimiyle eğitilir ve sınanır.

## Sonuçları okuma

`final_metrics.csv`: tam ve güven filtresi uygulanmış accuracy, balanced accuracy,
AUC, Brier, çoğunluk ve süreklilik referansları, örnek sayıları ve kapsam.
Yön referansı geçmiş momentum; oynaklık referansı mevcut oynaklık durumudur.
`accuracy_lift_ci_*` bu süreklilik referansına göre doğruluk farkının blok bootstrap
%95 aralığıdır. Parite başına çoklu karşılaştırma düzeltmesi içermez; tekil pozitif
aralık doğrulanmış ticari avantaj anlamına gelmez.

`cost_sensitivity.csv`: yön modeli için 0 / 1,5 / 3 pip toplam maliyet ve sabit
eşiklerin duyarlılığı. Finaldeki en güzel hücreyi seçerek model doğrulanmış sayılmaz.
Toplam baz puan portföy getirisi değildir; spread değişimi, swap, kayma ve kaldıraç yoktur.

`feature_group_lift.csv`: diğer grupların fiyat özelliklerine göre katkısı.
`feature_importance.csv`: EURUSD final örneğinde açıklama amaçlı permütasyon önemi;
sonradan özellik seçimi için kullanılmaz. İlişkili özellikler tekil önemi paylaşabilir.
`final_predictions.csv.gz`: tüm final tahminleri; `nonoverlap` sütunu ana ölçüm
örneklerini belirtir. `data_quality.json`: dahil/dışlanan pariteler ve onarımlar.

EURUSD, GBPUSD ve USDJPY'nin 2023+ dönemleri önceki araştırmada görülmüştü.
Bu üçü yeni kör test değildir. Diğer pariteler bu turnuvanın seçiminden sonra
kontrol edilir; ortak para birimleri nedeniyle pariteler bağımsız deney sayılmaz.

## Uygulamada anlamlı kullanım

Yön modeli maliyet sonrası tutarlı katkı göstermeden al/sat filtresine bağlanmaz.
Oynaklık modeli referansları geçerse, önce demo ortamında risk uyarısı, pozisyon
boyutu araştırması ve stop mesafesi değerlendirmesine yardımcı olabilir.
Oynaklık doğruluğu yön doğruluğu veya işlem kazanma oranı değildir.
Gerçek entegrasyon öncesinde mevcut TREND/RANGE girişleri üzerinde ayrı
meta-etiketleme, olasılık kalibrasyonu ve yeni ileri veriyle doğrulama gerekir.

Kaynaklar: [zaman sıralı doğrulama](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html),
[dengeli doğruluk](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html).
