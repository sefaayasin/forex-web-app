# Küçük net kârı önce alabilir miydi?

Aynı geçmiş fiyatlar ve aynı 19 GİR kararı. Yeni yön modeli veya yeni sinyal üretilmedi.
Her hedef için açık işlem çakışması yeniden hesaplandı; küçük hedef erken kapanırsa sonraki sinyal alınabilir.
Stop eski plandaki stop; üst süre 4 saat. Maliyet hedef mesafesine eklendi.

| Net hedef | İşlem | Hedef | Stop | Süre sonu | Net kâr/zarar | Ortalama net sonuç |
|---|---:|---:|---:|---:|---:|---:|
| 1 pip | 13 | 13 | 0 | 0 | 13 / 0 | +1.00 pip |
| 3 pip | 12 | 12 | 0 | 0 | 12 / 0 | +3.00 pip |
| 5 pip | 11 | 10 | 1 | 0 | 10 / 1 | +3.78 pip |
| $10 / $100 planlanan stop | 12 | 12 | 0 | 0 | 12 / 0 | +10.00 $ |
| $50 / $100 planlanan stop | 11 | 6 | 3 | 2 | 7 / 4 | -0.82 $ |

Dolar satırları örnek $100 stop bütçesine göre matematiksel ölçeklemedir; gerçek hesabın getirisi değildir.
Lot adımı, teminat, kur çevrimi ve değişen spread dahil edilmedi; broker kotasyonu yerine BID OHLC kullanıldı.
Daha yüksek spread/komisyon ve ek 0,5 pip kayma sonuçları results.json içinde ayrıca gösterilir.
Örnek çok küçük ve daha önce incelenmiş veridir. Yüksek hedefe ulaşma oranı, zarar etmeme garantisi vermez.
Sinyalin sonradan doğru yönde gitmesi yerine, küçük net hedefe stop öncesi ulaşması ölçüldü.
