# Kanıta dayalı yöntem araştırması — 9 Ekim 2026

Önceki negatif indikatör taramaları Forex'te kâr edilemeyeceğini kanıtlamaz.
Bir makalenin konu başlığından türetilen basit kuralın başarısızlığı da
makalenin yöntemini çürütmez. Önceki seans çalışması Londra açılışını inceledi;
Londra referans kurunun belirlenmesinden sonraki davranışı incelemedi.

## Öncelikli bulgular

1. [Krohn, Mueller ve Whelan, Bank of Canada, 2021](https://www.bankofcanada.ca/wp-content/uploads/2021/10/swp2021-48.pdf):
   Fixing çevresindeki dolar hareketi. Maliyet bölümü (PDF s.35–38) ve Tablo IX
   incelendi. Tam gösterge spreadleriyle çoğu sonuç negatif; yarı spread
   senaryosunda bazı Avrupa pencereleri pozitif. Bu ucuz işlem varsayımı
   kullanıcının hesabında doğrulanmalı. Çalışmanın veri sonu 2019; aşağıdaki
   uyarlama orijinal stratejinin birebir tekrarı değildir.
2. [Osler, New York Fed, 2001](https://www.newyorkfed.org/research/staff_reports/sr125.html):
   Destek/direnç çevresindeki dönüşler ve kırılım sonrası ivme için emirlerin
   yuvarlak fiyatlarda kümelenmesine dayanan mekanizma. Bu, rastgele bir
   osilatör eşiğinden farklıdır. Günümüzde masraf sonrası uygulanabilirliği
   veya otomatik 30 dolar kazancı kanıtlanmış sayılmaz.
3. [Osler, New York Fed, 2002](https://www.newyorkfed.org/research/staff_reports/sr150.html):
   Stop emirlerinin tetiklendiği seviyelerde hızlanan fiyat hareketleri;
   etkiler saat ölçeğinde görülür. Mum verisinden stop emirlerinin gerçek
   yerlerini bildiğimizi varsayamayız.
4. [Menkhoff ve diğerleri, BIS, 2011](https://www.bis.org/publications/working-paper-366-currency-momentum-strategies):
   Para birimleri arasında momentum getirisi bulgusu; maliyet ve uygulanabilirlik
   kısıtları var. Uzun dönemli para birimi portföy bulgusu doğrudan 15 dakikalık
   işlem sinyaline aktarılmamalı.
5. [CME FX Market Profile](https://www.cmegroup.com/trading/fx/cme-fx-market-profile-tool.html?itm_content=fx_2_mkt_profile_tool_122020):
   Emir defteri derinliği, hacim ve spread verisi mevcut. Bu bir veri kaynağıdır,
   kazanç kanıtı değildir. Spot mum veya tick sayısı tüm piyasanın gerçek
   alış/satış emir akışı diye kullanılamaz.

## Bu turda yapılan somut kontrol

`research_fixing.py`: Londra 16:15'te EURUSD ve GBPUSD alış; yaz/kış saati
otomatik. 1 ve 4 saat çıkış; 2016–2019, 2020–2022, 2023–2026 dönemleri ayrı.
Parametre seçilmedi, bütün sonuçlar `summary.csv` içinde. Sabit süreli çıkışın
pip sonuçları ile 30 USD hedef / 20 USD planlanan stop uyarlaması ayrı tutuldu.

2023 sonrası GBPUSD dört saat sabit süreli çıkışın ortalama brüt hareketi
**+0,816 pip**, varsayılan toplam maliyeti yaklaşık **1,307 pip**, neti
**-0,491 pip**. Bu fark gerçek hesabın masrafını ölçmeyi gerekli kılar; maliyeti
keyfî düşürerek kâr üretmek doğru olmaz. Aynı dönemin hedef/stop uyarlaması
929 işlemde **-0,256 USD/işlem**. EURUSD de bu uyarlamada doğrulanmadı.

İlk 15 dakika atlanıyor ve bekleme pencereleri makaleden farklı; orijinal
ön-fix bacağı uygulanmadı. Dolayısıyla sonuç yalnızca bu uyarlamayı ölçer.
Zaman çıkışlı pip analizinde koruyucu stop yoktur; 1.000 USD hesap için
uygulanabilir pozisyon büyüklüğü veya zarar sınırı sonucu değildir. Sabit
risk dolarları broker lot, teminat ve fiyat kayması doğrulaması içermez.

## Uygulama için araştırma kararı

Önce gerçek broker maliyeti ve uygulanabilir lot adımı ölçülmeli. Ardından
fixing stratejisinin asıl zaman pencereleri ve destek/direnç çevresindeki
olaylar için önceden belirlenmiş kurallar karşılaştırılmalı. Öğrenen model
bu olaylardan hangilerinin net kâr ihtimali taşıdığını öğrenmeli; aynı mumdan
türetilmiş çok sayıda indikatörün oylarını çoğaltmak yeni veri sağlamaz.

İlk model hedefi: net beklenen getiri, hedefe stop öncesi ulaşma olasılığı,
örnek sayısı ve tahmin belirsizliği. Hiçbir aday canlı işlem izni almadı.
Geçmiş veriler önceki araştırmalarda görüldü; ileri kayıt doğrulaması ayrıca
gereklidir. Kaynak, kod ve fiyat dosyalarının hash bilgileri raporlarda saklı.
