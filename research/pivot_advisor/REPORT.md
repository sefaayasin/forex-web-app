# Pivot Trader v2 — geçmiş veri testi

Kurallar değiştirilmeden 7 paritede tarandı. Model eğitimi 2023 öncesi; test 2024-10-01–2026-09-11.

Karar her 5 dakikalık mum kapanışında. Oluşan 15M mumun sonradan gerçekleşen kapanışı kullanılmadı.
Giriş sonraki 5M mum açılışında; aynı paritede açık işlem varken yeni işlem sayılmadı.
Plan 4 saat sonunda kapanır. Aynı mumda stop ve hedef varsa stop kabul edildi; stop boşlukları kötü fiyattan yazıldı.

| Yön | Giriş kararı | İşlem | 1s yön doğru | 4s yön doğru | Hedef | Stop | Süre sonu | Net kazanma | Net pip/işlem |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Tümü | 19 | 11 | %31.6 (6/19) | %55.6 (10/18) | 0 | 6 | 5 | %36.4 | -7.52 |
| LONG | 4 | 2 | %75.0 (3/4) | %100.0 (3/3) | 0 | 1 | 1 | %50.0 | -5.75 |
| SHORT | 15 | 9 | %20.0 (3/15) | %46.7 (7/15) | 0 | 5 | 4 | %33.3 | -7.91 |

## Sınırlar

- Bu veri Yahoo/broker canlı akışının kaydı değildir; yerel Dukascopy fiyatlarıyla aynı kodun tekrar oynatılmasıdır.
- Takvimde yalnız planlı yüksek etkili haber saatleri kullanıldı; actual/forecast/surprise okunmadı. Tarihsel takvim sürümleri doğrulanmadı.
- Spread ve komisyon saatlik tarihsel profil tahminidir; her işlem anının gerçek alış/satış kotasyonu yoktur.
- Stres sonucu iki kat maliyet ve ek 0,5 pip kayma içerir. Swap dahil değildir; fiyatlar USD kazancına çevrilmez.
- 1s/4s yön ölçümü tüm GİR kararlarını kapsar; tekrarlanan kararlar bağımsız işlemler değildir.
- İşlem sonucu ayrı ve çakışmasızdır. Eksik 5M yolları ve geçersiz sonraki açılışlar sayılmadı; audit.json sayıları içerir.
- Tarihsel test görülmüş veridir; gerçek ileri test / %70 başarı doğrulaması değildir.

Parite ayrıntıları, gün/saat, karar fiyatı, stop, hedef ve sonuçlar aynı klasördeki CSV dosyalarında.

## Tekrarsız işlemlerde yön

Aşağıdaki oranlarda her paritede yalnız aynı anda tek açık işlem sayılır; tekrar GİR kararları çıkarılmıştır.

| Yön | İşlem | 1 saat sonra yön doğru | 4 saat sonra yön doğru |
|---|---:|---:|---:|
| LONG | 2 | 2/2 (%100.0) | 2/2 (%100.0) |
| SHORT | 9 | 3/9 (%33.3) | 4/9 (%44.4) |

## Son LONG işlem örnekleri

Son tarihten geriye beş tekrarsız LONG işlemi; iyi/kötü sonuca göre seçilmedi.

| İstanbul saati | Parite | Giriş | Stop | Hedef | Çıkış | Net pip |
|---|---|---:|---:|---:|---|---:|
| 03.10.2024 20:35 | USDCHF | 0.85229 | 0.85116 | 0.85440 | 4 saat sonu | +1.50 |
| 23.04.2026 22:05 | USDJPY | 159.680 | 159.561 | 159.856 | Stop | -12.99 |
