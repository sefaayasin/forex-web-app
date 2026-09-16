# Sinyal sayımı — 16 Eylül 2026

EURUSD, EURCHF ve AUDCAD yerel 15 dakika arşivlerinin son 60 takvim günü
incelendi. Kullanılan veri 14 Temmuz–11 Eylül 2026 aralığında, parite başına
4.225 ham mum içeriyor. Bu inceleme kullanıcının canlı Yahoo oturumunun
birebir tekrarı değildir.

Ortak varsayımlar: Londra seansı, 1,5 pip maliyet, lot başına 10 USD pip değeri,
10.000 USD başlangıç, %0,5 risk, 16 mum bekleme, en fazla 24 mum pozisyon süresi.
Trend: eşik 60, düzeltme + tepki, tüm varsayılan teknik filtreler açık,
1,5 ATR, 1,5 R hedef, hibrit stop ve 1 R başabaş taşıma.
Yatay strateji: orta banda dönüş, maliyet sonrası en az 1 R hedef alanı.
Bunlar karşılaştırma varsayımlarıdır; broker maliyet ölçümü değildir.

## Trend stratejisi

| Parite | Yön/skor geçen mum | Yapı/giriş modeli sonrası | Tüm filtreler sonrası açılan işlem |
|---|---:|---:|---:|
| EURUSD | 1.232 | 14 | 0 |
| EURCHF | 1.781 | 15 | 4 |
| AUDCAD | 1.453 | 7 | 1 |

## Yatay strateji

| Parite | Bant dönüş tetiği | Uyumsuzluk filtreleri sonrası | Seans ve bekleme sonrası | Açılan işlem |
|---|---:|---:|---:|---:|
| EURUSD | 83 | 68 | 34 | 1 |
| EURCHF | 57 | 48 | 19 | 1 |
| AUDCAD | 71 | 66 | 24 | 0 |

Son adım hedef alanı, maliyet sonrası risk/getiri ve geçerli lot koşullarını
birlikte içerir. Ekrandaki yeni sayım hedef/risk koşulunu ayrıca gösterir.

Sayılar sıralı elemedir. Açık pozisyon varken yeni giriş aranmaz; bir aşamada
elenen mumlar sonraki aşamaya gitmez. Bu nedenle sayılar bağımsız filtre
katkısı veya bir filtre kaldırıldığında elde edilecek işlem sayısı değildir.
Özellikle yüksek eleme, filtrenin gereksiz olduğunu tek başına göstermez.

Eski ve sayaç eklenmiş kod, bu üç paritede iki motor için karşılaştırıldı:
işlem, bakiye eğrisi ve performans tabloları aynı kaldı. Eşikler ve işlem
kuralları bu değişiklikte gevşetilmedi.

## Arayüz düzeltmeleri

- Piyasa kararsızken aktif motor seçilememesi test eksikliği sayılmıyor.
- Ana kanıt özeti aktif motorun sonucunu kullanıyor; yatay test ile ayrı
  normal trend testinin işlem sayıları aynı kanıt gibi gösterilmiyor.
- Sıfır veya az işlem içeren tamamlanmış test, yapılmamış testten ayrılıyor.
- Son sekiz motor testi oturum içinde parite/ayar anahtarıyla saklanıyor.
  Bunlar kayıtlı test sonuçlarıdır; yeniden hesaplama Planı Kontrol Et ile yapılır.
- Geçmiş testte neden az işlem var bölümü her iki motorun sıralı sayımını ve
  gerçek veri aralığını gösteriyor.
