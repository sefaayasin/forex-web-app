# Sabitleme (fixing) saatleri etrafında dolar dönüşü — sonuç

5 Ekim 2026. Sonuç: dört pencerenin hiçbiri geçmedi. Makaledeki örüntü bizim verimizde de var. Ama 2020 sonrasında işlem başına 0,6–0,8 pipe inmiş ve ~1–1,5 pip maliyeti karşılamıyor.

## Soru

Krohn, Mueller & Whelan (Bank of Canada SWP 2021-48; Journal of Finance 2024) 1999–2019 verisinde her yıl şunu buldu: Dolar Tokyo (09:55 Tokyo) ve ECB (14:15 Frankfurt) sabitlemelerinden önce değer kazanıyor, Tokyo ve Londra (16:00 Londra) sabitlemelerinden sonra değer kaybediyor. Mekanizma, bankaların sabitlemedeki dolar talebine karşı önceden korunma yapması.

Makalede etki pencere başına ~2,5–3,5 baz puan; bu EURUSD için ~3–4 pip. Ölçülen EURUSD maliyeti ise ~1 pip. Soru şuydu: makalenin hiç görmediği 2020 sonrasında bu, maliyetten sonra kazandırıyor mu?

## Kurgu

- **Pencereler** (New York saati, yaz saati dahil):
  - Tokyo öncesi: 17:00 → Tokyo sabitlemesi
  - Tokyo sonrası: sabitleme → 02:00
  - ECB öncesi: 02:00 → 08:15
  - Londra sonrası: 11:00 → 17:00
- **İşlem:** Her pencere bir işlem. Önce penceresinde dolar alınıyor, sonra penceresinde dolar satılıyor.
- **Pariteler:** EURUSD, GBPUSD, AUDUSD, NZDUSD, USDJPY, USDCAD, USDCHF; Dukascopy 5M.
- **Maliyet:** Giriş ve çıkış saatindeki ölçülen medyan spreadlerin ortalaması + 0,7 pip komisyon.
- **Ana dönem:** 2020–2026. Her pencere için %98,75 aralık sıfırın üstünde olmalı ve 2020–2022 ile 2023–2026'nın ikisinde de net artıda olmalı.
- **Uygulama kontrolü:** 2008–2019'da makaledeki yönler görülmeli. Görüldü: yabancı para Tokyo öncesi −1,5, Tokyo sonrası +1,6, ECB öncesi −1,5, Londra sonrası +0,8 pip.
- Cuma günü Londra sonrası penceresinin çıkışı piyasa kapanışına denk geldiği için o işlemler kurala göre atlandı.
- Kurallar sonuçlardan önce protocol.json dosyasına yazıldı.

## Sonuçlar (2020–2026, işlem başına pip)

| Pencere | İşlem | Maliyet öncesi | Net | %98,75 aralık | Kazanma | Geçti mi |
|---|---:|---:|---:|---|---:|---|
| Tokyo öncesi (dolar al) | 11.352 | +0,17 | −3,43 | [−4,07, −2,75] | %50,2 | Hayır |
| Tokyo sonrası (dolar sat) | 11.776 | +0,65 | −0,81 | [−1,64, +0,09] | %50,9 | Hayır |
| ECB öncesi (dolar al) | 11.753 | +0,77 | −0,63 | [−1,99, +0,61] | %50,9 | Hayır |
| Londra sonrası (dolar sat) | 9.363 | −0,42 | −3,97 | [−5,26, −2,60] | %48,9 | Hayır |

- **Temiz iki pencere** (Tokyo sonrası, ECB öncesi): yön makaleyle aynı ama etki makaledekinin yaklaşık dörtte biri. En yakın tek parite EURUSD, ECB öncesi: maliyet öncesi +0,73, maliyet 1,00, net −0,27 pip.
- **17:00 New York'a değen iki pencere:** Burada sonuçlar gün sonu spread sıçramasından (rollover, spread 3–7 kat) bozuluyor. Veri alış (bid) fiyatı olduğu için o anda bütün pariteler aşağı sapıyor. Bu pencereler gerçekçi işlem pencereleri değil.
- Makale de son yıllarında etkinin azaldığını yazıyordu. Yayımdan sonra arbitraj sermayesinin etkiyi küçültmesi beklenen bir şey.

## Uygulamaya

Geçmediği için uygulamaya işlem paneli eklenmedi.

## Dosyalar

- protocol.json — önceden yazılan kurallar
- results.json — parite ve yıl bazında bütün sayılar
- ../../research_fix_reversal.py
