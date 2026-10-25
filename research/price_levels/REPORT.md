# Yuvarlak fiyat seviyesi olayları — 9 Ekim 2026

Kaynak gerekçesi: [Osler, New York Fed](https://www.newyorkfed.org/research/staff_reports/sr125.html).
Bu çalışma makalenin tekrarı değildir; gerçek emir konumları olmadan 50/100 pip
seviyelerini vekil olarak kullanan sabit bir olay tanımıdır.

Önceki kapanıştan alt/üst seviye belirlenir. Kapanış seviyeyi 0,1 ATR aşarsa
kırılım; seviye dışına uzanan mum içeri kapanır ve dönüş yönünde gövde bırakırsa
dönüş sinyali oluşur. Mum kapanmadan sinyal verilmez; önceki mumdan 15 dakikadan
fazla boşluk varsa olay atlanır. Çelişkili sinyal işleme dönüşmez.

Üç parite, dört olay kuralı, iki bekleme süresi, üç risk bütçesi: 72 ayar.
Hedef net 30 USD, risk 20/50/100 USD, stop 3 ATR. Seçim yalnızca 2016–2018
ve 2019–2022 dönemleriyle, her birinde en az 100 işlem şartıyla yapılır.
Kontrol 2023 sonrası; geçmiş daha önce incelendiğinden kör test değildir.

| Parite | Seçilen olay | Süre | İşlem | Ortalama net USD |
|---|---|---:|---:|---:|
| EURUSD | 50 pip kırılımı | 4 saat | 1.214 | -0,194 |
| GBPUSD | 50 pip kırılımı | 1 saat | 2.286 | -0,732 |
| USDJPY | 50 pip kırılımı | 4 saat | 1.670 | -0,460 |

Hepsinde geliştirme seçimi 20 USD planlanan risk kullandı. Toplam 5.170 işlem
ayrı strateji kayıtlarıdır; ortak hesap portföyü değildir. GBPUSD/USDJPY sabit
1.000 USD hesap senaryolarında bütçe tükendiği için tüm geçmiş işlemler gerçek
hesapta sürdürülebilir sayılmaz; raporlarda karşılanabilir işlem sayıları var.

EURUSD, toplam ücret 0,5 pip varsayılınca +0,200 USD/işlem verir; 1 pipte
-0,194 USD olur. Bu maliyet indirimini gerçek hesap bilgisi olmadan varsayamayız.
Ayrıca mevcut ücret profiliyle yıllık net ortalamalar 2023 +0,495; 2024 +1,269;
2025 -1,155; 2026 -1,706 USD'dir. Bu, geçmişteki pozitif dönemin genellenemediğini
gösterir. 2026 arşivi 11 Eylül'de biter. Düşük maliyet yeterli doğrulama değildir.

0/0,5/1/2 pip duyarlılık testlerinde hedef mesafeleri ve tüm işlem yolları
yeniden hesaplandı. Ücretsiz işlem yalnızca teşhis senaryosudur. Günlük
bootstrap, işlem başına ortalama yerine günlük işlem ortalamalarına uygulanır;
bu iki ortalama farklı ağırlıklandırılır. Hiçbir aday araştırma ölçütlerini geçmedi.

Kontroller: Gelecekteki son 1.000 mum kaldırılınca geçmiş sinyaller aynı kaldı.
Sentetik yükseliş/düşüş kırılımı ve dönüş örnekleri geçti. Ham işlem sayısı ve
net ortalamalar raporlarla karşılaştırıldı. Sonuçlar uygulamanın yeni indikatör
araştırma bölümünde, yıllık kırılımla birlikte gösterilir.

Çalıştırma: `python -X utf8 research_price_levels.py`.
Her paritenin `selection.json` dosyası bütün geliştirme ayarlarını;
`report.json` dosyası maliyet senaryolarını, kaynak hashlerini ve yılları içerir.
Broker lot/teminat gerçekleşmesi modellenmedi; gap zararları planlanan stopu aşabilir.
