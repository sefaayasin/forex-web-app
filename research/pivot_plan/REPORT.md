# Pivot trader planı — kutunun gösterdiği her planı uygulasaydın — sonuç

20 Ekim 2026. Sonuç: iki hipotez de geçmedi. Ders kitabı pivot planı maliyetten önce başabaş, maliyetten sonra işlem başına yaklaşık −2,3 pip kaybettiriyor. Rastgele kaydırılmış çizgilerle kurulan planlar da aynı sonucu veriyor. Kutu anasayfaya, kullanıcı istediği için yine eklendi; altında bu sonuç tek satırda yazıyor.

## Soru

Kullanıcı GBPJPY'de "Geç + Dönüş var" görünce SHORT girdi ve zarar etti. Nerede durup nerede kâr alacağını bilmiyordu. Bunun üzerine anasayfada bütün görüşleri okuyup bir pivot trader gibi "LONG / SHORT / BEKLE, şuradan gir, stop şurası, hedef şurası" diyen tek bir kutu istedi. Soru: kutunun gösterdiği her planı uygulasaydın ne olurdu?

## Kural (forex_pivot_plan.pivot_plan; kutu ve test aynı fonksiyonu kullanıyor)

- **Seviyeler:** Önceki hafta içi günün mumundan P, R1, R2, S1, S2 (research/pivot_stops ile aynı). Bunlara klasik R3 = H + 2(P − L) ve S3 = L − 2(H − P) eklendi.
- **Yön:** Fiyat P'nin üstündeyse LONG, altındaysa SHORT.
- **Giriş:** Fiyatın gerisindeki en yakın seviye. LONG'da fiyatın altındaki, SHORT'ta üstündeki seviye. Geri çekilmede limit emir.
- **Hedef:** Fiyatın önündeki en yakın seviye. R3'ün üstünde ya da S3'ün altında seviye kalmadıysa plan BEKLE der.
- **Stop:** Giriş seviyesinin 1 ATR ötesi (15M ATR14, gerçek aralığın basit ortalaması).
- **Filtre:** Hedef stoptan yakınsa plan BEKLE der.

## Kurgu

- **Veri:** 28 parite (EURZAR hariç), 2 Ocak 2008 – 11 Eylül 2026. Seviyeler Dukascopy günlük mumlarından, fiyat yolu 15 dakikalık mumlardan.
- **Karar anları:** Hafta içi her gün 00, 04, 08, 12, 16 ve 20:00 UTC'de kapanan 15M mum. Aynı paritede önceki plan bekliyor ya da açıkken yeni plan kurulmadı.
- **Uygulama:**
  - Limit emir 4 saat geçerli. Fiyat girişe gelmezse işlem yok.
  - Dolan işlemde stop ya da hedef, hangisi önce gelirse.
  - Dolduğu mumda yalnız stop kontrol edildi. Hem stopa hem hedefe değen mum stop sayıldı (temkinli).
  - İkisi de olmazsa UTC gün sonunda kapanışta çıkıldı.
- **Maliyet:** Ölçülen spread (dolumun New York saatine göre) + 0,7 pip komisyon.
- **Sahte çizgiler:** Bütün seviyeler önceki günün aralığının ±0,25 katı kaydırıldı, aynı kurallar uygulandı.
- **İstatistik:** Takvim günlerini blok olarak yeniden örnekleyen bootstrap (2.000 tekrar). İki hipotez olduğu için %97,5 aralık kullanıldı (Bonferroni).
- Bütün kurallar sonuç hesaplanmadan önce protocol.json dosyasına yazıldı.

## Sonuçlar

| | Plan | BEKLE | Dolma oranı | Dolan | Hedef | Stop | Gün sonu | Brüt pip | Net pip | Net R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gerçek seviyeler | 490.889 | %21 | %47 | 181.832 | %14,2 | %72,5 | %13,4 | +0,19 | −2,27 | −0,283 |
| Sahte seviyeler | 983.413 | %21 | %45 | 347.622 | %14,3 | %72,4 | %13,4 | +0,05 | −2,41 | −0,292 |

Medyan stop 9,2 pip, medyan maliyet 2,1 pip.

| Hipotez | Değer | %97,5 aralık | Geçti mi |
|---|---:|---|---|
| H1: maliyetten sonra kazandırıyor (net R > 0) | −0,283 R | [−0,297, −0,270] | Hayır |
| H2: gerçek seviyeler sahtelerden iyi (brüt R farkı > 0) | +0,012 R | [−0,005, +0,027] | Hayır |

| Dönem | Net pip | Hedef | Stop |
|---|---:|---:|---:|
| 2008–2015 | −2,08 | %14,5 | %71,7 |
| 2016–2023 | −2,32 | %13,8 | %72,5 |
| 2024–2026 | −2,65 | %14,2 | %74,4 |

- **LONG ve SHORT:** LONG brüt −0,04 pip, SHORT brüt +0,44 pip; ikisi de maliyetten sonra eksi.
- **Pariteler:** 28 paritenin hiçbirinde net sonuç artı değil. GBPUSD: dolan planların %12,6'sı hedefe, %72,2'si stopa gitti, net −0,67 pip. GBPJPY: %14,3 hedef, %71,1 stop, net −1,51 pip.

## Yorum

- **Plan neden çoğunlukla stopa gidiyor?** Hedef bir sonraki pivot seviyesi; stop ise 1 ATR. Hedef çoğu zaman stoptan birkaç kat uzak (filtre en az 1 kat istiyor). Fiyat rastgele yürüseydi de uzak hedefe daha nadir, yakın stopa daha sık gidilirdi. Brüt sonucun sıfır çıkması bunu gösteriyor: oranlar mesafelere göre beklenen kadar, çizgiler bir avantaj eklemiyor.
- **Kaybı maliyet belirliyor.** 9 piplik stopta 2,1 pip maliyet, riskin yaklaşık %20'si.
- **Kutunun asıl faydası seviyeleri önceden belirlemek.** GBPJPY olayında kural SHORT demezdi: fiyat P'nin üstündeydi, hatta R3'ü geçmişti, yani kutu "BEKLE" derdi. Ama bu, kuralın para kazandırdığı anlamına gelmiyor.

## Uygulamaya

Anasayfaya "📐 Pivot Trader" kutusu eklendi (Genel Yorum ile İşlem Planı arasında). İçinde şunlar var:

- yön, giriş, stop, hedef ve planın bozulduğu nokta,
- diğer görüşler tek satırda (Özet kartı, Fırsatlar, Pariteler, ML),
- Geç + dönüş uyarısı,
- 60 dakika içinde yüksek etkili haber varsa "BEKLE".

Altında bu testin sonucu tek satırda yazıyor. Haber ve Geç + dönüş satırları plan kuralını değiştirmiyor, yalnızca uyarı. Bu iki uyarı ayrıca test edilmedi; Geç + dönüş için research/freshness'a bakılabilir.

## Dosyalar

- protocol.json — önceden yazılan kurallar
- results.json — bütün sayılar (dönem, yön ve parite dökümü)
- ../../forex_pivot_plan.py — plan kuralı ve ATR
- ../../research_pivot_plan.py — `build` ve `evaluate`
- ../../test_forex_pivot_plan.py, ../../test_research_pivot_plan.py — birim testleri
