# Haber anında 15M + 5M yön durumu — araştırma sonucu

26 Eylül 2026. Sonuç: test edilen dört versiyonun hiçbiri önceden yazılan koşulları geçmedi. Mekanik haliyle "15M ve 5M aynı yönü gösterirken haberde o yöne gir" kuralı, en düşük maliyette (1,5 pip) bile ortalama zarar etti. Canlı uygulamanın işlem kuralları değiştirilmedi.

## Soru ve kurallar

Kullanıcı gözlemi: 15M/5M "al" durumunu takip edip haber anını değerlendirdiğimizde işlemlerin çoğu kazandı (işlem sayısı bilinmiyor). Bu gözlemin mekanik bir versiyonu test edildi.

- **Olaylar:** NFP (2008'den beri, BLS takvim kuralıyla, 08:30 New York) ve FOMC (2013'ten beri planlı çarşamba kararları, 14:00 New York).
- **Pariteler:** EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD. Veri yerel 5M ve 15M arşivlerinden geldi.
- **Yön durumu:** Uygulamanın skor fonksiyonu kullanıldı. Son kapanmış 15M ve 5M skorlarının ikisi de +25 veya üstündeyse LONG (uygulamadaki "Alım Yönlü" eşiği), ikisi de −25 veya altındaysa SHORT. Diğer durumlarda işlem yok.
- **Önce:** Habere kapanmış mumlarla karar verilir, habere son fiyattan girilir.
- **Sonra:** Karar haberden 15 dakika sonra, tepki mumları da dahil edilerek verilir.
- **İşlem:** Stop, 15M ATR14'ün 1,5 katı. Hedef, stop mesafesinin 1,5 katı. En fazla 4 saat tutulur. Stop ve hedef aynı mumdaysa stop önce sayılır.
- **Kontrol:** Aynı kural, bir hafta önceki aynı gün ve saatte, yani haber olmadan.
- **İstatistik birimi:** Olay. Pariteler aynı USD haberine tepki verdiği için bağımsız sayılmadı; her olayda sinyal veren paritelerin ortalama R'si alındı.

Kurallar ve başarı koşulu, hiçbir sonuç hesaplanmadan önce protocol.json dosyasına yazıldı. Sonradan ayar değiştirilmedi.

## Ana sonuçlar

R = stop mesafesi cinsinden net sonuç. −0,25 R, riske edilen tutarın çeyreğinin işlem başına ortalama kaybedilmesi demektir. 1,5 R hedefle maliyetsiz başabaş kazanma oranı %40'tır.

| Olay | Versiyon | Olay | İşlem | Kazanma (1,5 pip) | Ort. R, 1,5 pip | Ort. R, 3 pip [%95 GA] | Haber yokken, 3 pip |
|---|---|---:|---:|---:|---:|---:|---:|
| NFP | Önce | 190 | 781 | %42,3 | −0,11 | −0,26 [−0,39, −0,13] | −0,30 |
| NFP | Sonra | 191 | 888 | %39,5 | −0,13 | −0,24 [−0,35, −0,12] | −0,22 |
| FOMC | Önce | 103 | 418 | %30,6 | −0,36 | −0,50 [−0,66, −0,34] | −0,37 |
| FOMC | Sonra | 97 | 442 | %40,5 | −0,16 | −0,27 [−0,46, −0,09] | −0,35 |

Dönemlere göre (3 pip): NFP Önce −0,28 / −0,23, NFP Sonra −0,18 / −0,32, FOMC Önce −0,57 / −0,45, FOMC Sonra −0,04 / −0,46 (2008–2018 / 2019–2026). 5 pip haber spreadinde bütün versiyonlar −0,37 ile −0,69 R arasında.

## Değerlendirme

- **Haber hareketi büyütüyor ama yönü bu kural bilmiyor.** FOMC sonrası girişlerde hedefe ulaşma oranı %37, haber yokken %12. Buna rağmen kazanma oranı başabaş civarında kaldı ve maliyetten sonra sonuç eksi.
- **Haber yokken de sonuç eksi.** Kontrol grubunda da her versiyon zararda. Bu, projedeki önceki bulgularla uyumlu: 15M/5M yön skorunun kendisinin maliyet sonrası avantajı yok.
- **FOMC öncesi yön, açıklamadan sonra ters dönme eğiliminde.** Açıklamadan sonraki 1 saatte fiyat, 15M/5M yönünün ortalama 3,8 pip tersine gitti (haber yokken −0,1 pip). Bu, sonuçlar görüldükten sonra fark edilen bir gözlem. Ayrı bir önceden kayıtlı test olmadan strateji sayılmamalı.

Bu sonuç, kullanıcının kazandığı işlemleri geçersiz kılmaz. Gösterdiği şey, kazancı mekanik kuralın açıklamadığıdır. Avantaj varsa büyük ihtimalle kurala sığmayan yerlerdedir: hangi haberin seçildiği, tepkinin nasıl okunduğu, hangi işlemden kaçınıldığı ve çıkış yönetimi. Bunu ölçmenin yolu, her işlemi bu bilgilerle kaydedip sonucunu takip etmektir.

## Sınırlar

- Sadece NFP ve FOMC var. CPI, perakende satışlar, ECB veya BoE kararları için geçmiş açıklama saati elimizde yok.
- Tatil ve hükümet kapanması yüzünden kayan NFP tarihleri modellenmedi. Açıklama mumu olağan aralığın 3 katından küçükse olay elendi (30 olay, listesi results.json dosyasında). Bu elemeyle 2016-07-08 ve 2020-11-06 gibi birkaç gerçek ama tepkisi sönük NFP günü de dışarıda kaldı. Bu, örneği büyük hareketlere doğru hafifçe kaydırır.
- Fiyatlar spread içermeyen arşiv fiyatları. Gerçek haber spreadi ve kayması çoğu zaman 1,5 pipten fazladır, bu da sonuçları daha kötüleştirir.
- Skor, uygulamanın backtest için kullandığı vektörel skordur; canlı ekrandaki skorla birebir aynı değildir. Sabit stop/hedef kullanıldı; kullanıcının çıkış yönetimi farklı olabilir.
- Veri eksikliği yüzünden bazı parite-olaylar atlandı (paritelere göre 6–152 arası, results.json dosyasında).

## Tekrar çalıştırma

`python -X utf8 research_news_scan.py`

Yerel data/historical_5m ve data/historical_15m arşivleri gerekir. İnternetten veri çekilmez. events.csv olay listesini ve zaman kontrolünü, trades.csv bütün işlemleri, summary.csv özet istatistikleri, results.json karar koşullarını içerir.
