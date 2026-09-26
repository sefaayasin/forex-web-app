# Gerçek spread ölçümü — sonuç

27 Eylül 2026. Dukascopy'nin alış (bid) ve satış (ask) fiyatlarından 28 paritenin spreadi ölçüldü. Dukascopy bir ECN bankası. Buradaki değerler komisyon hariç ham spreaddir ve "iyi bir hesapta en az bu kadar ödersin" anlamında bir alt sınırdır. Komisyonsuz "standart" perakende hesaplarda spread genelde daha geniştir.

## Yöntem

- **Genel profil:** Haziran–Ağustos 2026 arasından seçilmiş 10 iş günü (haftanın her günü ikişer kez). Her dakikanın kapanışında satış eksi alış fiyatı alındı; sadece iki tarafta da işlem olan dakikalar sayıldı. Parite başına yaklaşık 14 bin dakika.
- **Haber anları:** Aynı dönemdeki 3 NFP (2 Temmuz, 7 Ağustos, 4 Eylül) ve 1 FOMC (29 Temmuz). Açıklama saatinin tick verisi EURUSD, GBPUSD ve USDJPY için incelendi.
- 628 dosyanın 628'i indi. Uygulama profili New York saatine göre tutuyor. Seanslar ve gün sonu rollover'ı (New York 17:00) yaz/kış saatine göre UTC'de kaydığı için bu, kışın da doğru saati verir.

## Parite bazında maliyet

- **Maliyet** = Londra–New York saatlerindeki medyan spread + tipik ECN komisyonu (0,7 pip).
- **Maliyet payı** = maliyet ÷ (stop + maliyet). Stop olarak, 2023 sonrası 15M/5M sinyallerinin gerçek medyan stopu (1,5 × ATR14, 15M) kullanıldı.
- **Rollover** = gün sonu kapanış saati (New York 17:00).

| Parite | Spread medyan | Rollover | Maliyet | Eski varsayılan | Medyan stop | Maliyet payı |
|---|---:|---:|---:|---:|---:|---:|
| USDJPY | 0,4 | 1,4 | 1,1 | 1,0 | 15,3 | **%7** |
| EURJPY | 0,8 | 3,0 | 1,5 | 2,0 | 16,2 | %9 |
| AUDJPY | 0,6 | 2,2 | 1,3 | 2,0 | 12,6 | %9 |
| GBPJPY | 1,6 | 4,9 | 2,2 | 2,0 | 20,5 | %10 |
| GBPUSD | 0,7 | 1,9 | 1,3 | 1,0 | 11,1 | %11 |
| EURUSD | 0,3 | 1,1 | 1,0 | 1,0 | 8,5 | %11 |
| NZDJPY | 0,8 | 3,4 | 1,4 | 2,0 | 11,3 | %11 |
| CADJPY | 1,0 | 3,6 | 1,7 | 2,0 | 11,8 | %13 |
| CHFJPY | 2,0 | 5,6 | 2,6 | 2,0 | 17,9 | %13 |
| GBPAUD | 2,1 | 5,7 | 2,7 | 2,0 | 18,1 | %13 |
| EURAUD | 2,0 | 4,1 | 2,6 | 2,0 | 15,5 | %14 |
| USDCHF | 0,8 | 2,4 | 1,5 | 1,0 | 7,8 | %16 |
| USDCAD | 1,1 | 2,8 | 1,7 | 1,0 | 8,8 | %16 |
| AUDUSD | 0,9 | 1,8 | 1,6 | 1,0 | 7,7 | %17 |
| EURCAD | 1,7 | 4,5 | 2,3 | 2,0 | 11,0 | %17 |
| NZDUSD | 1,0 | 2,1 | 1,6 | 1,0 | 7,2 | %18 |
| EURNZD | 3,4 | 6,4 | 4,0 | 2,0 | 17,6 | %19 |
| GBPCAD | 2,6 | 6,6 | 3,2 | 2,0 | 13,9 | %19 |
| GBPCHF | 1,6 | 4,0 | 2,3 | 2,0 | 9,6 | %19 |
| EURGBP | 0,7 | 1,8 | 1,3 | 2,0 | 5,1 | %20 |
| EURCHF | 0,9 | 2,3 | 1,6 | 2,0 | 6,2 | %21 |
| GBPNZD | 5,0 | 8,8 | 5,4 | 2,0 | 20,5 | %21 |
| AUDCAD | 1,8 | 3,8 | 2,4 | 2,0 | 8,4 | %22 |
| AUDCHF | 1,4 | 4,1 | 2,0 | 2,0 | 6,7 | %23 |
| AUDNZD | 1,9 | 4,5 | 2,5 | 2,0 | 7,6 | %25 |
| CADCHF | 1,4 | 7,6 | 2,0 | 2,0 | 6,0 | %25 |
| NZDCAD | 2,2 | 6,5 | 2,8 | 2,0 | 8,2 | %25 |
| NZDCHF | 1,5 | 5,1 | 2,2 | 2,0 | 6,1 | %26 |

Değerler pip cinsinden. "Eski varsayılan", uygulamanın maliyet ayarında daha önce kullandığı sabit değer.

## Haber anında spread (tick verisi, ilk 60 saniye)

| Olay | Parite | 5 dk önce medyan | İlk 60 sn en yüksek | İlk 60 sn medyan | Normale dönüş |
|---|---|---:|---:|---:|---:|
| NFP 2 Tem | EURUSD / GBPUSD / USDJPY | 0,3 / 0,8 / 0,5 | 2,4 / 4,5 / 4,7 | 0,4 / 1,0 / 1,0 | 1 / 0 / 0 sn |
| NFP 7 Ağu | EURUSD / GBPUSD / USDJPY | 0,4 / 0,8 / 0,5 | 7,0 / 5,8 / 9,5 | 0,5 / 1,0 / 1,0 | 1 / 0 / 3 sn |
| NFP 4 Eyl | EURUSD / GBPUSD / USDJPY | 0,4 / 0,7 / 0,6 | 3,5 / 6,9 / 10,3 | 0,5 / 0,9 / 1,1 | 1 / 2 / 6 sn |
| FOMC 29 Tem | EURUSD / GBPUSD / USDJPY | — | 8,0 / 7,5 / 13,1 | 0,7 / 1,1 / 0,7 | — |

"Normale dönüş", spreadin açıklama öncesi medyanın 2 katının altına indiği ana kadar geçen süre. FOMC saat başında (18:00 UTC) açıklandığı için "5 dk önce" penceresi ayrı bir saat dosyasında kalıyor ve ölçülmedi.

## Değerlendirme

- **Maliyeti belirleyen spread değil, spreadin stopa oranı.** JPY'li pariteler ve GBPUSD/EURUSD'de maliyet stopun yaklaşık %7–11'i. CHF, CAD ve NZD çaprazlarında %20–26. EURGBP ve EURCHF'nin spreadi düşük ama stopları çok dar (5–6 pip), bu yüzden onlar da pahalı.
- **Rollover'dan uzak durmak gerekir.** New York 17:00'de (TR yazın 00:00, kışın 01:00) spread 2–7 katına çıkıyor.
- **Seanslar arasında fark küçük.** Asya saatlerinde spread Londra–New York saatlerinden sadece biraz geniş.
- **Haber anında spread birkaç saniyelik bir sıçrama yapıyor.** Açıklamanın ilk saniyelerinde spread normalin 5–25 katına çıkıyor (EURUSD 7 pipe, USDJPY 13 pipe kadar), ama çoğunlukla birkaç saniyede normale dönüyor. Açıklama anında piyasa emriyle girmek, bu en kötü fiyatı yakalama riski taşır. Birkaç saniye beklemek maliyeti büyük ölçüde normale indirir; ancak fiyat o saniyelerde çoktan hareket etmiş olabilir.
- **Önceki araştırmalardaki maliyet varsayımı çoğu parite için iyimserdi.** 1,5 pip varsayımı 28 paritenin 20'sinde ölçülen maliyetin altında kaldı; çaprazlarda gerçek sonuçlar daha kötü olurdu. Bu, o testlerin "avantaj yok" sonucunu güçlendirir. Meta-etiketleme modelinin JPY ve GBP'li pariteleri seçmesi de maliyet payının orada düşük olmasıyla uyumlu.

## Uygulamaya yansıyanlar

- **Maliyet ayarının varsayılanı:** Artık o saatte ölçülen medyan spread + 0,7 pip komisyon. Eski varsayılan sabit 1,0 (majörler) / 2,0 (diğerleri) pipti. Broker'ın MT5 veya CSV verisi varsa, yine önce broker'ın kendi spreadi kullanılıyor.
- **Sinyal sekmesi:** "Maliyet: X pip = planlanan kaybın %Y'si" notu eklendi. %15'i aşarsa uyarı olarak gösteriliyor, ve o saatin medyan ile yüksek spreadini de gösteriyor.
- **Tarama tablosu:** Şu anki saatin medyan spreadini gösteren bir sütun eklendi.

## Sınırlar

- Ham ECN spreadi; broker'ın markup'ı ve gerçek komisyonu farklı olabilir.
- 10 örnek gün ve 4 haber anı; ekonomik takvimin yoğun olduğu dönemlerde ya da kriz anlarında spread daha geniş olabilir.
- Dakika kapanışındaki spread, dakika içindeki sıçramaları göstermez; haber anı için bu yüzden tick verisi kullanıldı.

## Tekrar çalıştırma

`python -X utf8 measure_spreads.py`

İnternet gerekir. Dukascopy sunucusu yavaş olduğu için ilk çalıştırma saatler sürebilir; indirilen dosyalar research/spreads/cache altında saklanır (git dışı) ve sonraki çalıştırmalar oradan okur. Çıktılar: data/ml/spread_profile.json (uygulama okur) ve research/spreads/results.json.
