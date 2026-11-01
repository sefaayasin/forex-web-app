# ML dönüş çıkışı: %50 çizgisi, %52 güven, iki saat teyit

1 Kasım 2026. **Sonuç: geçmedi, canlı bot değişmedi.** Saatlik düzende ML dönüş çıkışı işlemlerin yalnız %1,9'unda devreye giriyor. Üç alternatif de sonucu 0,002 R'den az oynatıyor; protokoldeki +0,01 R şartının çok altında.

## Soru

Bot saat başında ML yönü pozisyonun tersine geçince çıkıyor; %50,1'lik ters okuma bile yetiyor. Kapanış sebebi olarak ters yönde en az %52 güven ya da iki saat üst üste teyit aranmalı mı? Kurallar sonuçlardan önce `protocol.json`'a yazıldı.

## Kurgu

- Bot portföy motoru, 28 parite, 2023-01-01 – 2026-09-10, 5M adım, sabit risk görünümü (12,5 $).
- Taban: `K1` düzeni. Saatlik ML, ML güveni ≥ 0,55, ulaşma ≥ %65, maliyet riskin ≤ %10'u, 0,3 R net sabit hedef, yarım stop, takas, sepet takibi.
- Kontrol: E0, K1'i birebir üretti (8.520 işlem, −6.490,60 $, −0,0649 R).
- Çıkış değişince sonraki girişler de değiştiği için fark, E0 ile ortak girişlerde eşleştirildi. Ortak pay %99'un üstünde.
- Aralık: haftalık blok bootstrap, %95, 2.000 tekrar.

## Sonuçlar

| Varyant | İşlem | Net $ | 2023–24 R | 2025–26 R | Ort. kayıp R | R ≤ −0,75 | ML çıkışı | 60 dk içinde yeniden giriş |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 mevcut (%50) | 8.520 | −6.491 | −0,0645 | −0,0654 | −0,537 | 38 | 163 | 2.729 |
| E1 ters güven ≥ %52 | 8.511 | −6.512 | −0,0643 | −0,0663 | −0,543 | 41 | 84 | 2.720 |
| E2 iki saat teyit | 8.505 | −6.429 | −0,0634 | −0,0656 | −0,544 | 41 | 41 | 2.718 |
| E3 ML çıkışı yok | 8.506 | −6.396 | −0,0627 | −0,0657 | −0,546 | 41 | 0 | 2.719 |

Ortanca tutma süresi dört varyantta da 15 dakika.

Ortak girişlerde E0'a göre fark:

| Varyant | Değişen işlem | Fark R/işlem | %95 aralık | 2023–24 | 2025–26 |
|---|---:|---:|---:|---:|---:|
| E1 | 97 | −0,0001 | −0,0011 … +0,0009 | +0,0008 | −0,0009 |
| E2 | 159 | +0,0007 | −0,0005 … +0,0019 | +0,0017 | −0,0003 |
| E3 | 158 | +0,0012 | +0,0000 … +0,0024 | +0,0024 | −0,0001 |

## Yorum

- **Etki yok denecek kadar küçük.** Hedef ve yarım stop çoğu işlemi ilk saat başından önce kapatıyor. Saat başındaki kontrol 8.520 işlemin 163'üne denk geliyor.
- **Daha az erken çıkış daha iyi sonuç değil.** E1 ML çıkışını yarıya indirdi, yine de ortalama kaybı ve büyük kayıp sayısını biraz artırdı, net sonucu değiştirmedi.
- **Yeniden giriş ayrı bir konu.** İşlemlerin yaklaşık üçte biri aynı paritede, önceki kapanıştan en çok 60 dakika sonra açılıyor. Çıkış kuralı bunu değiştirmiyor.
- **E3 de aday değil.** Aralığı sıfıra değiyor, 2025–26'da fark yok ve +0,01 R şartını geçmiyor.

## Karar

Protokol şartı sağlanmadı: E1 ve E2 iki dönemde de +0,01 R iyi değil. Saatlik düzende çıkış kuralı değişmedi.

## Sınırlar

- **`--ml-every 5` bu çalışmayla ölçülmedi.** Motor saat başına hizalı ML kullanıyor; kayan mumların geçmiş simülasyonu yok. 27 Ekim–30 Ekim canlı verisinde 5 dakikalık kontrol yön değişimini 4,5 kat artırmıştı. O düzende çıkış etkisi çok daha büyük olabilir, bu rapor onun güvenli olduğunu göstermez.
- Tarihler önceki araştırmalarda incelendi; yeni kör test değildir.
- Motor anlık spread sıçramasını ve kaymayı modellemiyor. Bütün varyantlar eksi kaldı.

## Tekrar üretme

```powershell
python -X utf8 research_exit_confirm.py
python -X utf8 -m unittest test_research_exit_confirm test_research_bot_backtest
```

Sonuçlar `results.json` dosyasında. Bu çalışma canlı bota, emirlere veya günlüğe dokunmadı.
