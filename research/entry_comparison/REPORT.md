# Giriş modeli ve seans karşılaştırması

16 Eylül 2026. Yerel arşivde 14 Temmuz–11 Eylül 2026, 15 dakikalık verinin son 60 takvim günü. Kullanıcının Yahoo ekranındaki 25 Haziran–16 Eylül aralığının birebir tekrarı değildir.

Her varyant ayrı 10.000 USD hesap, %0,5 işlem riski, sabit 1,5 pip maliyet ve lot başına 10 USD pip değeriyle test edildi. RSI/MACD ve diğer filtreler aynı kaldı. Tüm Gün, seans kısıtını kaldırır; Londra/New York kesişimi değildir. Gece spreadlerinin değişmesi bu sabit maliyet modelinde temsil edilmiyor.

| Parite | Model | Seans | İşlem | Net sonuç ($) | En büyük düşüş (%) | Ortalama R |
|---|---|---|---:|---:|---:|---:|
| EURUSD | Düzeltme + Tepki | Londra | 0 | 0.00 | 0.00 | — |
| EURUSD | Düzeltme + Tepki | Tüm Gün | 1 | -50.00 | 0.50 | -1.000 |
| EURUSD | Trend + Yapı | Londra | 4 | -100.61 | 1.01 | -0.505 |
| EURUSD | Trend + Yapı | Tüm Gün | 8 | -231.31 | 2.31 | -0.584 |
| EURCHF | Düzeltme + Tepki | Londra | 4 | -54.37 | 1.09 | -0.271 |
| EURCHF | Düzeltme + Tepki | Tüm Gün | 4 | -54.37 | 1.09 | -0.271 |
| EURCHF | Trend + Yapı | Londra | 11 | -209.45 | 2.94 | -0.383 |
| EURCHF | Trend + Yapı | Tüm Gün | 16 | -339.25 | 3.70 | -0.430 |
| AUDCAD | Düzeltme + Tepki | Londra | 1 | -50.00 | 0.50 | -1.000 |
| AUDCAD | Düzeltme + Tepki | Tüm Gün | 3 | -149.25 | 1.49 | -1.000 |
| AUDCAD | Trend + Yapı | Londra | 4 | -180.53 | 1.81 | -0.909 |
| AUDCAD | Trend + Yapı | Tüm Gün | 7 | -327.09 | 3.27 | -0.948 |

## Sonuç

Bu örnekte Trend + Yapı daha fazla işlem oluşturdu; ancak her üç paritede Londra ve Tüm Gün sonuçları negatif kaldı. Mevcut düzeltme/tepki modeli de bu örnekte pozitif sonuç göstermedi. Hiç işlem yapmayan varyantın sıfır sonucu başarılı strateji kanıtı değildir.

Örnekler 0–16 işlem aralığında: stratejilerin genel performansı hakkında güçlü bir çıkarım yapılamaz. Bu karşılaştırmadan sonra bir varyant seçmek, ayrı veri üzerinde doğrulama yerine geçmez. Son %30 işlem ölçümü JSON/CSV içinde araştırma amaçlı bulunur; bağımsız doğrulama iddiası değildir.

Varsayılan giriş modeli ve seans değiştirilmedi. Sayım artık üst zaman dilimi yapısı ile giriş modelini ayrı gösteriyor. Aşağıdaki sayılar yalnız mevcut Düzeltme + Tepki / Londra koşulunda sıralı elenmedir:

| Parite | Yön/skor | Üst zaman dilimi yapısı sonrası | Düzeltme/tepki sonrası |
|---|---:|---:|---:|
| EURUSD | 1232 | 238 | 14 |
| EURCHF | 1781 | 501 | 15 |
| AUDCAD | 1443 | 189 | 7 |

## Tekrar çalıştırma

Proje kökünde `python -X utf8 compare_entry_models.py` çalıştırılır. Yerel historical_15m, historical_1h ve historical_4h arşivleri gerekir. İnternetten veri çekilmez; canlı ayarlar değiştirilmez. Ham sonuçlar results.csv ve details.json dosyalarına yazılır.

Düşüş hesabı başlangıç bakiyesini içerir; ilk işlemdeki kayıp atlanmaz. Bakiye eğrisi gerçekleşmiş sonuçları kullanır; açık pozisyonların mum içi değer kaybını ölçmez.
