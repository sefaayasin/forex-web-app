# Basit saatlik strateji — araştırma sonucu

16 Eylül 2026. Sonuç: hiçbir parite önceden belirlenen araştırma koşullarını geçmedi. Bu model canlı uygulamaya eklenmedi; mevcut işlem kuralları değiştirilmedi.

## Kurallar ve dönemler

Saatlik EMA50/EMA200 kesişimi; kapanan mumdan sonraki açılışta giriş. 1,5 ATR stop, stop mesafesinin 1,5 katı hedef, en fazla 48 işlem mumu taşıma. Tek pozisyon, tüm saatler, işlem başına maliyet dahil %0,5 risk. RSI, MACD, düzeltme/tepki veya üst zaman dilimi filtresi yok.

2010–2018 geliştirme, 2019–2022 doğrulama, 2023–2025 son test olarak ayrıldı. Her dönem ayrı 10.000 USD bakiye ile başladı. Önceki veriler yalnız geçmişe bakan göstergeleri hazırlamak için kullanıldı. Dönemler arasında pozisyon taşınmadı. Parametre araması veya sonuç sonrası ayar değişikliği yapılmadı.

Kurallar protocol.json dosyasına ilk sonuçtan önce yazıldı. Son test dönemi başka proje araştırmalarında daha önce yer aldığından kapalı, tamamen görülmemiş dış doğrulama seti olduğu iddia edilmiyor.

## Ana sonuçlar — 1,5 pip toplam işlem maliyeti

| Parite | Dönem | İşlem | Net getiri (%) | En büyük düşüş (%) | Ortalama net R |
|---|---|---:|---:|---:|---:|
| EURUSD | 2010–2018 | 310 | 21.24 | 7.03 | 0.128 |
| EURUSD | 2019–2022 | 150 | -13.51 | 14.63 | -0.190 |
| EURUSD | 2023–2025 | 100 | -4.62 | 6.69 | -0.091 |
| EURCHF | 2010–2018 | 226 | -18.20 | 20.13 | -0.175 |
| EURCHF | 2019–2022 | 146 | -8.77 | 13.66 | -0.123 |
| EURCHF | 2023–2025 | 118 | -13.27 | 14.68 | -0.238 |
| AUDCAD | 2010–2018 | 328 | -19.81 | 19.95 | -0.131 |
| AUDCAD | 2019–2022 | 134 | 1.79 | 4.37 | 0.030 |
| AUDCAD | 2023–2025 | 122 | -14.33 | 18.91 | -0.250 |

## Maliyet duyarlılığı — 2023–2025

| Parite | 1,5 pip net getiri (%) | 3 pip net getiri (%) |
|---|---:|---:|
| EURUSD | -4.62 | -7.40 |
| EURCHF | -13.27 | -16.68 |
| AUDCAD | -14.33 | -17.07 |

## Değerlendirme

EURUSD geliştirme döneminde pozitifti, ancak doğrulama ve son test dönemleri negatif kaldı. AUDCAD doğrulama döneminde küçük bir pozitif sonuç gösterdi; güven aralığı sıfırı içeriyor ve maliyet 3 pipe çıkınca sonuç negatife dönüyor. Son testte üç parite de zarar etti.

Doğrulama ve son testte en az 60 işlem, pozitif ortalama net R, sıfırın üzerinde %95 bootstrap alt sınırı ve üç parite için düzeltilmiş p ≤ 0,05 koşulları birlikte arandı. Ayrıca son testin 3 pip maliyette pozitif kalması istendi. Hiçbiri bu koşulları sağlamadı. Bu sonuç daha fazla işlem üretmenin tek başına stratejik avantaj oluşturmadığını gösterir; tüm basit stratejilerin başarısız olduğuna dair bir iddia değildir.

## Veri kontrolü ve sınırlar

Arşivler değiştirilmedi. Strateji sonuçları hesaplanmadan önce 41 mumda en çok 0,1 pip OHLC aralık tutarsızlığı tespit edildi. Bellekte High/Low, kaydedilmiş Open/Close değerlerini kapsayacak kadar genişletildi. Bu değişiklik data_amendment.json içinde kayıtlı; daha büyük hatalar çalışmayı durdurur. Ham dosya SHA-256 özetleri ve yıllık mum sayıları results.json içinde bulunur.

| Parite | Ham mum | Küçük aralık düzeltmesi | 2 saatten büyük veri aralığı |
|---|---:|---:|---:|
| EURUSD | 114653 | 13 | 979 |
| EURCHF | 95347 | 18 | 835 |
| AUDCAD | 110884 | 10 | 954 |

Veri aralıkları hafta sonları ve eksik kayıtları birlikte içerir; tam kapsama iddiası yoktur. İki saatten eski giriş sinyalleri atlandı. Açık pozisyonlar boşluk boyunca taşınabilir; kötü stop açılışları açılış fiyatıyla hesaba katıldı. Aynı mumda stop ve hedef görülürse stop önce kabul edildi. Hedef boşlukları hedef fiyatından, süre sonu çıkışları mum açılışından hesaplandı.

Maliyetler sabit varsayımdır; gerçek bid/ask, swap/finansman ve gün içi değişen spread yoktur. En büyük düşüş gerçekleşmiş bakiye üzerinden hesaplanır; açık pozisyonların mum içi zararını içermez. Her parite ayrı hesap gibi test edildi; sonuçlar ortak portföy getirisi olarak toplanamaz.

## Tekrar çalıştırma

`python -X utf8 research_simple_baseline.py`

Yerel data/historical_1h arşivleri gerekir. Ağ erişimi yoktur. summary.csv özet sonuçları, results.json istatistikleri/veri kontrollerini, parite ve dönem CSV dosyaları tüm işlemleri içerir. 2026 verileri dönem performansına dahil edilmedi.
