# 4 saatlik oynaklık modeli — son test sonucu

26 Eylül 2026. Sonuç: model önceden yazılan dört ölçütten birini geçemedi ve uygulamaya eklenmedi. Yayındaki 72 saatlik oynaklık modeli değişmedi.

**Tam veriyle yeniden çalıştırma (aynı gün):** Bu testte kullanılan yerel 1H arşivinde birçok çapraz paritede bütün işlem haftaları eksikti. Test aynı kurallarla, 15M'den yeniden kurulan tam saatlik veriyle tekrarlandı (data/ml/vol4h_rebuilt). Dengeli isabet %58,2, son-4-saat tahmini %56,4, fark **1,8 puan**. İlk ölçüt yine geçilmedi; karar değişmedi. Aşağıdaki sayılar ilk çalıştırmaya aittir.

## Neden denendi

72 saat, 15M/5M işlem yapan biri için uzun bir vade. Turnuvanın geliştirme testinde 4 saatlik adaylar daha yüksek AUC gösterdi (0,77'ye karşı 0,70). Basit tahmine göre farkları da daha büyüktü. Ama hiçbiri 2023 sonrası son teste girmemişti.

## Kurgu

- **Seçim:** Sadece geliştirme sonuçlarına bakıldı (rankings.csv). Aday havuzu: 4 saat vadesi, COT hariç özellik setleri ve uygulamanın yükleyebildiği scikit-learn modelleri. Kazanan: "context" özellik seti + hist_boost. Geliştirmede dengeli isabeti %61,5; "son 4 saat neyse o devam eder" tahmini %57,3.
- **Güven eşiği:** Turnuvanın kuralıyla, geliştirme verisinde seçildi ve 0,50 çıktı. Eşik yükseldikçe model sadece "sakin" diyor ve dengeli isabeti %50'ye düşüyor.
- **Son test:** 28 parite. Eğitim 2008–2022, test Ocak 2023–Eylül 2026. Toplam 73.362 örtüşmeyen 4 saatlik pencere.
- **Protokol eki (son testten önce yazıldı):** Modelin özelliklerinde günün saati var ve 4 saatlik oynaklığın büyük kısmı seanstan gelir. Bu yüzden "sadece saate bakan" tahmin de kıyasa ve başarı ölçütüne eklendi (protocol_amendment.json).

## Sonuç

Değerler 28 paritenin eşit ağırlıklı ortalamasıdır.

| Tahmin | Dengeli isabet | AUC |
|---|---:|---:|
| Model (hist_boost) | %58,2 | 0,745 |
| Son 4 saat neyse o devam eder | %56,8 | — |
| Sadece günün saati | %53,3 | 0,667 |
| Saat + son oynaklık (sade lojistik, bilgi amaçlı) | %55,9 | 0,723 |

| Ölçüt | Sonuç |
|---|---|
| Dengeli isabette son-4-saat tahminine göre en az 2 puan fark | **1,4 puan — geçmedi** |
| Paritelerin en az yarısında isabet farkının güven aralığı sıfırın üstünde | Geçti (28/28) |
| Emin tahminlerde son-4-saat ve çoğunluk tahminini geçmek | Geçti (%79,3; karşılaştırmalar %64,5 ve %78,5) |
| Sadece saate bakan tahmini geçmek | Geçti (%58,2'ye karşı %53,3) |

İkinci ve üçüncü ölçütün kolay geçilmesinin sebebi, pencerelerin sadece %21,5'inin "yüksek oynaklık" olması. Son-4-saat tahmini sık yanlış alarm verdiği için düz isabette geride kalıyor. Model ise "hep sakin" diyen tahminden yalnızca 0,8 puan daha isabetli.

## Değerlendirme

- **Dengeli isabette model, 28 paritenin sadece 12'sinde son-4-saat tahmininden iyi.**
- **En iyi göründüğü paritelerde saati bilmek yetiyor.** EURGBP, GBPCAD ve GBPUSD'de yalnız saate bakan tahmin modelden daha iyi: %80,3 / %76,2 / %72,4'e karşı %76,0 / %70,9 / %67,6. Oradaki başarı büyük ölçüde Londra açılışını bilmekten geliyor.
- **Sıralama gücü var ama kanıtı yok.** AUC 0,745 fena değil ve 27 paritede sade lojistikten yüksek. Model "hangi saatte, hangi parite daha riskli" sıralamasında bilgi taşıyor olabilir. Ancak bu sonuçlara bakıldıktan sonra fark edildi. Kullanılmadan önce yeni, önceden yazılmış bir testle ve Eylül 2026 sonrası yeni veriyle doğrulanması gerekir.
- **Pratik anlamı:** Önümüzdeki 4 saatin oynaklığı büyük ölçüde seans saatinden ve son saatlerin hareketinden zaten tahmin edilebiliyor. Bu model buna anlamlı bir şey eklemiyor.

## Sınırlar

- Test tam kör değil: 2023 sonrası dönem başka hedeflerin son testinde kullanıldı; EURUSD, GBPUSD ve USDJPY'nin bu dönemi önceki araştırmalarda incelendi.
- Pariteler ortak para birimleri paylaştığı için bağımsız deney sayılmaz.
- Eşik 0,50 kaldığı için "emin tahmin" ölçütü tüm örneğe eşit.
- Kaydedilen 4 saatlik model dosyaları (data/ml/tournament/*_high_volatility_4h_research.joblib) yerelde durur. Git'e eklenmez ve uygulama tarafından kullanılmaz.

## Tekrar çalıştırma

`python -X utf8 research_vol4h.py select`, ardından `python -X utf8 research_vol4h.py final`. Sonuç dosyaları varken final aşaması yeniden çalışmaz. selected.json, final_metrics.csv ve summary.json bütün sayıları içerir.
