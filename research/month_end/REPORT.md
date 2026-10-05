# Ay sonu hisse korumasının Londra 16:00 sabitlemesine etkisi — sonuç

5 Ekim 2026. Sonuç: geçmedi. Önceden yazılan kural 2014–2026'da net −0,29 pip. Sonuçlara bakınca umut verici görünen bir alt kural (büyük hisse farkı olan aylar) hiç bakılmamış 2003–2007 verisinde doğrulanmadı: net −1,52 pip.

## Soru

Melvin & Prins (Journal of Financial Markets 22, 2015; veri 2004–2012) şunu gösterdi: Uluslararası hisse fonları döviz korumalarını ayın son günü Londra 16:00 sabitlemesinde ayarlıyor. Bir ülkenin hisse piyasası ay içinde iyi gittiyse, o ülkenin parası sabitlemeden önceki son bir saatte değer kaybediyor. %10'luk hisse artışı ~14 baz puan düşüş getiriyor. Soru şuydu: tahmin edilen yöne 15:00–16:00 Londra saatinde girmek, makalenin görmediği aylarda maliyetten sonra kazandırıyor mu?

## Kurgu

- **Sinyal:** R = yabancı endeks getirisi − S&P 500 getirisi. Önceki ay sonundan işlem gününden bir önceki iş gününe kadar, yerel para cinsinden. R > 0 ise yabancı para satılır, R < 0 ise alınır.
- **Endeksler:** Euro Stoxx 50, FTSE 100, ASX 200, NZX 50, Nikkei 225, TSX, SMI; ABD için S&P 500.
- **İşlem:** Ayın son iş günü, 15:00 → 16:00 Londra saati. 7 dolar paritesi, Dukascopy 5M, ölçülen spread + 0,7 pip komisyon.
- **Ana dönem:** 2014–2026. %95 aralık sıfırın üstünde olmalı ve 2014–2019 ile 2020–2026'nın ikisinde de net artıda olmalı.
- Kurallar ve hisse verisi sonuçlardan önce kaydedildi (protocol.json, equity_closes.csv).

## Sonuçlar

**Ana test** (2014–2026, 1.064 işlem, 193 ay sonu):

| | Maliyet öncesi | Net | Aralık | Kazanma |
|---|---:|---:|---|---:|
| Tümü | +1,11 | −0,29 | [−2,75, +1,94] | %51,6 |
| 2014–2019 | | +1,44 | | |
| 2020–2026 | | −1,84 | | |
| Makalenin döneminde (2008–2012, bilgi) | +2,84 | +1,44 | | %55,0 |

Geçmedi.

**Sonuçlara bakınca görülen ipucu ve doğrulaması:**

- **İpucu:** Hisse farkının en büyük olduğu üçte birlik dilimde (|R| ≥ %2,91) 2014–2026 neti +4,48 pip [+0,04, +8,73] ve kazanma %61 çıktı. Her alt dönemde artıdaydı. Aynı saat ayın diğer günlerinde düz kaldı (−0,09).
- **Neden doğrulama gerekti:** Eşik sonuçlardan seçildiği için bu bir bulgu değil, hipotezdi. Bu yüzden önce kurallar yazıldı (validation_protocol.json), sonra projede hiç kullanılmamış 2003–2007 ay sonu verisi Dukascopy'den indirildi. Euro bölgesi için DAX kullanıldı, çünkü Euro Stoxx 50 Yahoo'da 2007'den başlıyor.
- **Doğrulama sonucu:** 88 işlem, 43 ay sonu. Maliyet öncesi −0,16, net **−1,52 pip**, %90 aralık [−6,76, +3,54]. Geçmedi.
- **Yorum:** Büyük hisse farkı olan aylardaki pozitif sonuç büyük olasılıkla tesadüftü.

## Yorum

- Etki makalenin döneminde vardı (2008–2012 neti +1,44). Sonra zayıfladı ve 2020'den beri eksi.
- Bu, yayımlanan avantajların zamanla kaybolduğu genel örüntüyle uyumlu.
- Doğrulama adımı olmasaydı +4,5 pip gösteren alt kural uygulamaya konabilirdi. Hiç görülmemiş veride ise çalışmadı.

## Uygulamaya

Geçmediği için işlem paneli eklenmedi. Validation_protocol.json'daki ileri test (2026-09 sonrası ay sonları) istenirse `research_month_end_validation.py --forward` ile değerlendirilebilir. Ama A doğrulaması geçmediği için bunu takip etmek için güçlü bir sebep yok.

## Dosyalar

- protocol.json, validation_protocol.json — önceden yazılan kurallar
- results.json, validation_A_unseen_2003_2007.json — sayılar
- equity_closes.csv, equity_closes_2002_2008.csv — kullanılan hisse kapanışları
- ../../research_month_end.py, ../../research_month_end_validation.py
- ../../tick_downloader/download_month_end_days.js — 2003–2007 ay sonu verisinin indiricisi
