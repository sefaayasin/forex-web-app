# Ekonomik takvim kaynakları — hangisinin beklentisi daha isabetli?

30 Eylül 2026. **Sonuç:** Ücretsiz kaynaklar arasında beklentiyi (konsensüs) güvenilir biçimde daha iyi tutturan bir kaynak yok. Önceden yazılan karar kuralını hiçbir kaynak geçmedi. Uygulamanın takvim sayfası **ForexFactory**'yi kullanır, çünkü:

- En geniş kapsama sahip: yüksek önemli haberlerin %98'inde beklenti veriyor.
- Doğrulukta diğerlerinden geri değil.
- Uygulama zaten bu kaynağı kullanıyor ve sunucudan erişilebiliyor.

## Yöntem

Kurallar sonuçlardan önce [protocol.json](protocol.json) dosyasına yazıldı. Sonradan yapılan iki değişiklik de sonuç görülmeden eklendi:

- Investing indirmesi önem filtresine daraltıldı.
- Investing, sürekli engellendiği için (HTTP 429/403) ana karşılaştırmadan çıkarıldı.

Ayrıntılar:

- **Dönem ve kapsam:** Ekim 2024 – Eylül 2026; USD, EUR, GBP, JPY, CAD, AUD, NZD, CHF.
- **Referans olaylar:** ForexFactory'nin yüksek (ana set, 1.056 olay) ve orta (ikincil set, 754 olay) önemli, açıklanmış değeri olan haberleri.
- **Eşleştirme:** Aynı para birimi, 5 dakika içinde aynı saat ve aynı açıklanan değer.
- **Hata:** |açıklanan − beklenti|. Farklı birimleri karşılaştırabilmek için her haber serisinin tipik hatasına bölündü.
- **Karşılaştırma:** Kaynaklar ikişer ikişer, aynı olaylarda karşılaştırıldı: işaret testi (Holm düzeltmeli) ve gün bazlı bootstrap güven aralığı.
- **Nasdaq verisi:** Ekim 2024 – Mart 2026 arası indirildi; sonrası indirme durdurulduğu için eksik.

## Sonuçlar (yüksek önemli haberler)

| Kaynak | Beklenti verdiği olay oranı |
|---|---:|
| ForexFactory | %98 |
| FXStreet | %79 |
| Nasdaq | %38 (çoğunlukla ABD) |

| Karşılaştırma | Ortak olay | İlki daha yakın | Aynı | İkincisi daha yakın | Holm p | Ortalama ölçekli hata farkı [%95 GA] |
|---|---:|---:|---:|---:|---:|---|
| ForexFactory – FXStreet | 831 | 170 | 534 | 127 | 0,044 | −0,009 [−0,036, 0,018] |
| ForexFactory – Nasdaq | 400 | 7 | 385 | 8 | 1 | −0,005 [−0,017, 0,007] |
| FXStreet – Nasdaq | 324 | 48 | 216 | 60 | 0,58 | +0,003 [−0,038, 0,040] |

- **ForexFactory – FXStreet:** Beklentiler farklı olduğunda ForexFactory daha sık yakın çıkıyor (170'e 127, Holm p = 0,044). Ancak ortalama hata farkının güven aralığı sıfırı içeriyor; kural gereği bu bir "kazanan" sayılmadı.
- **ForexFactory – Nasdaq:** Olayların %96'sında beklenti birebir aynı.
- **Orta önemli haberler:** Hiçbir fark anlamlı değil (tüm Holm p ≥ 0,45).

## Investing (yan not, 12 hafta)

Investing yalnızca Ekim–Aralık 2024 arasında indirilebildi.

- Bu pencerede yüksek önemli 116 ortak olayın 105'inde ForexFactory ile aynı beklentiyi verdi.
- Nasdaq ile 47 ortak olayın 47'sinde aynıydı.
- Hiçbir karşılaştırma anlamlı değil.

Kısacası Investing, ForexFactory'den daha iyi bir beklenti sunmuyor; beklentiler neredeyse tamamen aynı.

## Yorum

Bu takvimlerin beklentileri büyük ölçüde aynı ekonomist anketlerinden türetiliyor. Bu yüzden aralarındaki fark çok küçük. Konsensüsü sistematik olarak geçen ücretsiz bir takvim kaynağı bulunmadı.

Belirli göstergeler için konsensüsten farklı tahminler üreten, ücretsiz ama tek göstergeye özel modeller var. Örnekler: Cleveland Fed'in enflasyon nowcast'i (TÜFE), Atlanta Fed GDPNow (ABD GSYH). Bunlar bu testin kapsamında değildi.

## Dosyalar

- `protocol.json`: kurallar ve değişiklikler
- `results.json`: tüm sayılar, yan pencere dahil
- `events.csv`: olay bazında hatalar
- `../../research_calendar_sources.py`: indirme ve analiz
- `fetch_investing.js`: Investing indirmesi (Playwright)
