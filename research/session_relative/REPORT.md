# Seans ve pariteler arası hareket testi

Yedi parite, altı kural, iki bekleme süresi ve üç risk bütçesi: 252 ayar.
Hedef net 30 USD, planlanan risk 20/50/100 USD, stop 3 ATR. Karar kapanmış
15 dakikalık mumdan, giriş sonraki 5 dakikalık mum açılışından hesaplanır.
Normal maliyet spread profili + komisyon; stres maliyeti bunun iki katıdır.

Kurallar Londra açılışı kırılımı, açılış aralığına dönüş, seans ortalamasına
dönüş, diğer altı paritenin dolar hareketini yakalama ve göreli hareketin
devamı/tersine dönüşüdür. Pariteler aynı zaman damgasında eşleşir; eksik
fiyatlar ileri taşınmaz. Londra saati yaz/kış saatini izler. Ortak simülatör
07–17 UTC giriş sınırı uygular. Her paritede gelecekteki son 1.000 mum
çıkarıldığında önceki sinyallerin değişmediği doğrulandı.

Seçim: 2016–2018 ve 2019–2022 dönemlerinin her birinde en az 100 işlem,
en zayıf dönemin stres netini en yüksek tutan ayar. Kontrol: 2023 sonrası.
Kontrol geçmişi önceden incelenmiştir; bağımsız kör test değildir.

| Parite | Kontrol işlemi | Ortalama net USD | İki kat masraf USD |
|---|---:|---:|---:|
| EURUSD | 155 | -1,68 | -2,55 |
| GBPUSD | 199 | -0,17 | -0,99 |
| USDJPY | 378 | -0,75 | -1,44 |
| AUDUSD | 102 | -0,69 | -2,18 |
| USDCAD | 688 | -1,58 | -3,12 |
| USDCHF | 295 | -1,37 | -2,71 |
| NZDUSD | 56 | -0,89 | -2,34 |

Seçilen ayarların tamamı 20 USD risk kullandı. Toplam 1.873 kontrol işlemi;
hiçbir yöntem araştırma ölçütlerini geçmedi. İşlem izni verilmedi.
USDCAD için sabit bütçeli 1.000 USD hesap 610 işlem sonrasında bir sonraki
20 USD planlanan riski karşılayamaz duruma geldi; bütün 688 işlemin ortalaması
ayrıca yöntem teşhisi için raporlanmıştır. Pariteler tek portföy gibi toplanmaz.

Araştırma gerekçeleri: [çoklu parite girdileri](https://arxiv.org/abs/1411.2153)
ve [gün içi mevsimsellik](https://arxiv.org/abs/1103.5664).
Bunlar burada kullanılan kuralların kârlılığını kanıtlamaz; makalelerin
yöntemleri birebir yeniden uygulanmadı.

Çalıştırma: `python -X utf8 research_session_relative.py`.
Ham kontrol işlemleri parite klasöründe, tüm geliştirme ayarları
`selection.json` dosyasında tutulur. Gerçek broker lot/teminat koşulları
modellenmemiştir; stop boşluklarında planlanan zarar aşılabilir.
