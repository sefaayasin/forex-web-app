# Mumlardan çizilen çizgiler (destek/direnç, yuvarlak sayı, trend çizgisi) — sonuç

28 Eylül 2026. Sonuç: altı hipotezin hiçbiri geçmedi. Çizgiler maliyetten önce bile fiyat hakkında, rastgele seçilmiş çizgilerden fazla bilgi taşımıyor. Uygulamaya çizgi tabanlı AL/SAT sinyali eklenmedi.

## Soru

Day trader'lar grafikte önceki tepe ve diplere yatay destek/direnç çizgileri çekiyor, yuvarlak sayıları işaretliyor ve trend çizgisi çiziyor. Fiyat bir çizgiden dönünce ya da çizgiyi kırınca alım/satım yapıyorlar. Bu çizgiler gözle değil de sabit kurallarla çizilirse:

1. Maliyetten sonra kazandırıyor mu?
2. Aynı kurallar, çizgi yerine rastgele fiyatlara çekilmiş sahte çizgilere uygulandığında sonuç daha mı kötü? Yani kazancı çizginin kendisi mi getiriyor?

**Literatür:**
- Osler (2000, New York Fed): Altı firmanın yayımladığı destek/direnç seviyelerinde gün içi trendler, rastgele seviyelere göre daha sık durmuş. Etki küçük ve kârlılık ölçülmemiş.
- Osler (2003): Büyük bir bankanın emir verisinde (Ağustos 1999 – Nisan 2000) kâr-al emirleri yuvarlak sayılarda, stop emirleri de yuvarlak sayıların hemen ötesinde yığılıyor. Bu, fiyatın yuvarlak sayılarda daha sık dönmesini ve geçilince hızlanmasını açıklıyor.
- Osler (2000) 1996–1998, Osler (2003) 1999–2000 verisini kullanıyor. Burada test edilen yılların hepsi yayın sonrası. (Protokolde iki çalışma için de "1996–1998" yazıyor; Osler (2003) için bu yanlış. Arka plan bilgisi olduğu için hiçbir kuralı etkilemiyor.)

## Kurgu

- **Veri:** 28 parite, 1 Ocak 2008 – 11 Eylül 2026. Çizgiler ve sinyaller Dukascopy'nin 15 dakikalık mumlarından, giriş ve çıkışlar 5 dakikalık mumlardan hesaplandı. Her karar yalnızca o an kapanmış mumları kullanıyor.
- **Tepe/dip:** Kendinden önceki 12 mumun (3 saat) hepsinden yüksek, sonraki 12 mumdan düşük olmayan mum tepedir; dip bunun tersidir. Tepe/dip ancak sonraki 12 mum kapandıktan sonra kullanılabiliyor.
- **Destek/direnç seviyesi:** Son 480 mumdaki (yaklaşık 5 işlem günü) tepe ve diplerden, birbirine 0,5 × ATR'den yakın en az iki tanesinin ortalaması. Grafikte her an ortalama 5,5–5,8 seviye oluyor; mumların %30–40'ı bir seviyeye değiyor.
- **Yuvarlak sayı:** Her 50 pip (örneğin 1,1000, 1,1050; 150,00, 150,50).
- **Trend çizgisi:** Son iki dipten geçen yükselen destek çizgisi ve son iki tepeden geçen düşen direnç çizgisi. İki nokta arasında hiçbir mum çizginin öbür yanında kapanmamış olmalı. Çizgi kırılınca ya da yeni bir tepe/dip oluşunca siliniyor. Toplamda yaklaşık 295 bin çizgi çizildi, parite başına 10.500.
- **Kurallar** (b = 0,25 × ATR):
  - **Dönüş:** Fiyat çizgiye yukarıdan geldi, çizgiye değdi (b payıyla) ve mum çizginin üstünde kapattı → al. Direnç için tersi → sat. Stop: çizginin ve mumun fitilinin b kadar ötesi.
  - **Kırılım:** Kapanış çizgiyi b kadar geçti → kırılım yönünde işlem. Stop: kırılan çizginin b kadar öbür tarafı.
  - Stop en az 0,5 × ATR. Hedef 1,5R. En fazla 4 saat. Aynı anda parite başına tek işlem. New York 17:00 (rollover) saatindeki girişler atlandı.
- **Maliyet:** Ölçülen medyan spread (girişin New York saatine göre) + 0,7 pip komisyon. Stres testi: iki katı.
- **Sahte çizgiler (plasebo):**
  - Destek/direnç ve trend çizgileri için: her gerçek çizginin 1,5 ve 3 ATR yukarı ve aşağı kaydırılmış dört kopyası.
  - Yuvarlak sayılar için: 10, 20, 30 ve 40 pip kaydırılmış ızgaralar (..10/..60, ..20/..70 gibi).
  - Sahte çizgiler gerçekleriyle aynı kurallarla işlem gördü.
- **İstatistik:** İşlem günlerini bloklar halinde yeniden örnekleyen bootstrap (2.000 tekrar). Altı hipotez test edildiği için eşik p ≤ 0,05 / 6 ≈ 0,0083.
- Bütün kurallar, parametreler ve ölçütler sonuç hesaplanmadan önce protocol.json dosyasına yazıldı. Hiçbir parametre sonuçlara bakılarak seçilmedi.

## Sonuçlar

| Hipotez | İşlem | Kazanan | Maliyet öncesi R | Net R | Sahte çizgi net R | Gerçek − sahte (%95 GA) | Medyan stop | Maliyet / risk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Destek/direnç — dönüş | 1.056.103 | %40,0 | −0,005 | −0,354 | −0,354 | −0,001 [−0,004, +0,002] | 7,5 pip | %22 |
| Destek/direnç — kırılım | 1.125.004 | %37,7 | −0,060 | −0,394 | −0,395 | +0,001 [−0,001, +0,003] | 7,7 pip | %22 |
| Yuvarlak sayı — dönüş | 983.442 | %40,3 | −0,003 | −0,284 | −0,285 | +0,002 [−0,001, +0,004] | 9,9 pip | %18 |
| Yuvarlak sayı — kırılım | 1.045.711 | %38,0 | −0,058 | −0,325 | −0,330 | **+0,005** [+0,003, +0,007] | 10,3 pip | %18 |
| Trend çizgisi — dönüş | 211.660 | %40,0 | +0,001 | −0,379 | −0,354 | −0,025 [−0,031, −0,018] | 7,0 pip | %23 |
| Trend çizgisi — kırılım | 172.264 | %37,8 | −0,057 | −0,420 | −0,391 | −0,028 [−0,035, −0,022] | 7,1 pip | %23 |

"Maliyet / risk" = maliyet ÷ (stop + maliyet), medyan.

| Ölçüt | Sonuç |
|---|---|
| Net R > 0 (p ≤ 0,0083) | 6 hipotezin 6'sında **geçmedi**; hepsinde −0,28 ile −0,42 arası |
| Sahte çizgilerden iyi (p ≤ 0,0083) | Sadece yuvarlak sayı kırılımında geçti (+0,005 R) |
| Stres maliyetinde net R > 0 | 6'sında da geçmedi |
| 2008–2013, 2014–2019, 2020–2026'nın her birinde net R > 0 | 6'sında da geçmedi; **19 yılın hepsinde eksi** |
| 28 paritenin en az 15'inde net R > 0 | 6'sında da geçmedi; **28 paritenin hiçbirinde artı değil** |
| En az 1.000 işlem | Hepsinde geçti |

## Değerlendirme

- **Dönüş kuralı maliyetten önce tam başabaş.** 1,5R hedefte başabaş kazanma oranı %40. Gerçekleşen oran %40,0–40,3 ve sahte çizgilerde de aynı. Yani fiyat, çizilen çizgide rastgele bir fiyattan daha sık dönmüyor.
- **Kırılım kuralı maliyetten önce bile kaybediyor (−0,06 R).** Sahte çizgilerde de aynı kayıp var. Bu yüzden sorun çizgide değil, "güçlü kapanışın arkasından girme" mekaniğinde: 15 dakikalık grafikte sert bir kapanıştan sonra fiyat biraz geri gelme eğiliminde. Projenin önceki 15M/5M kırılım sinyalleri de aynı sonucu vermişti.
- **Osler'in yuvarlak sayı bulgusu var, ama kullanılamayacak kadar küçük.** Yuvarlak sayı kırılımı sahte ızgaralardan anlamlı derecede iyi: maliyet öncesi +0,0036 R. 10 piplik stopta bu işlem başına yaklaşık 0,04 pip ediyor, maliyet ise medyan 2,2 pip. Etki maliyetin kabaca 60'ta biri.
- **Trend çizgileri sahtelerinden daha kötü.** Bunun sebebi gerçek trend çizgilerinde stopun biraz daha dar olması (7,0'a karşı 7,3 pip). Dar stopta aynı maliyet riskin daha büyük payı ediyor. Maliyet öncesi farkı anlamlı değil (dönüş) ya da hafif eksi (kırılım).
- **Kaybı maliyet belirliyor.** Maliyet riskin %18–23'ü. Kazanan her işlem için 1,5R'den maliyet düşüyor, kaybeden her işlemde de 1R'ye maliyet ekleniyor. Bu yüzden başabaş için kazanma oranının %40 değil yaklaşık %50 olması gerekirdi. Gerçekleşen oran %38–40.
- **Yıllar geçtikçe sonuç kötüleşiyor, ama avantaj kaybolduğu için değil.** Maliyet öncesi sonuç her yıl sıfır civarında (dönüş) ya da eksi (kırılım). Değişen şey stopun boyu: seviye stopları 2008'de medyan 14,7 pipti, 2024–2026'da 5–6 pip. Oynaklık düştükçe stoplar daralmış, aynı maliyet riskin daha büyük payını almış. Üstelik 2008–2012'de gerçek spreadler daha genişti; o yıllar burada olduğundan iyi görünüyor.
- **Grafikte neden "işe yarıyor" gibi görünüyor?** Sistem her an ortalama 5–6 seviye çiziyor ve mumların üçte biri bir seviyeye değiyor. Bu kadar çizginin olduğu bir grafikte her gün "tutmuş" bir çizgi bulunur. Ama aynı sayıda rastgele çizgi de aynı sıklıkla "tutuyor" (%40'a karşı %40).

## Sınırlar

- Test mekanik çizgileri ölçüyor, bir trader'ın gözle çizdiği çizgileri değil. Trader çizgiyi bağlama göre (trend, haber, seans) seçebilir. Ancak bu projede o bağlam filtreleri (15M/5M yön skoru, 4H/1H kuralı, ML modeli, haber anı) ayrıca test edildi ve hiçbiri avantaj göstermedi.
- Tek parametre seti denendi (12 mumluk tepe/dip, 5 günlük pencere, 1,5R, 4 saat). Sonuçlara bakıp parametre değiştirmek aşırı uyum olurdu. Maliyet öncesi sonuç bütün kurallarda ve sahte çizgilerde sıfır civarında olduğu için, başka bir ayarın riskin %20'si kadar maliyeti aşması beklenmez.
- Sadece 15 dakikalık grafik test edildi. 1H/4H'ten çizilen seviyeler, önceki günün en yüksek/en düşük seviyesi ve pivot noktaları test edilmedi.
- 2026'da ölçülen spread bütün yıllara uygulandı (eski yıllar için iyimser).
- Dukascopy fiyatları alış (bid) fiyatı. Açığa satışta stop aslında satış (ask) fiyatıyla, yani spread kadar erken tetiklenir; bu da sonucu biraz iyimser yapar.
- Sinyallerin %2'si 5M verisi eksik olduğu için, %4'ü rollover saatine denk geldiği için atlandı.

## Uygulamaya etkisi

- Çizgi tabanlı AL/SAT sinyali eklenmedi.
- Çizgileri grafikte yalnızca bilgi olarak göstermek ayrı bir karar; bu test o konuda bir şey söylemiyor, ama çizgilerin yön hakkında bilgi taşımadığını gösteriyor.

## Tekrar çalıştırma

```
python -X utf8 research_levels.py build      # işlemler research/levels/trades altına (git dışı), yalnız sayımlar yazdırılır
python -X utf8 research_levels.py evaluate   # sonuçlar; yalnız bir kez çalışır
```

Yerel data/historical_15m ve data/historical_5m arşivleri ile data/ml/spread_profile.json gerekir. Bütün sayılar results.json'da. Yıl ve parite dökümü by_year.csv ve by_pair.csv'de. Sayımlar ve atlanan sinyaller build.json'da. Birim testleri: `python -m unittest test_research_levels.py`.
