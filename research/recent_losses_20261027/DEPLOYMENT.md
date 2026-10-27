# 27 Ekim 2026 demo uygulaması

Kullanıcının onayıyla yeni demo işlemleri için uygulanan ayarlar:

- Hedefin stoptan önce gelmesine ilişkin tarihsel kontrol: en az %65.
- Ayrı ham ML yön güveni: en az %55, LONG ve SHORT için simetrik.
- İşlem başına risk bütçesi: sanal bakiyenin %1'i.
- Yeni işlemler: sabit hedef broker TP emrine yazılır; hedefte çıkış, mevcut yarım stop korunur.
- Saatlik ML dönüş çıkışı, süre sınırı, takas, sepet ve diğer giriş limitleri devam eder.

```powershell
python -X utf8 -u forex_bot.py --deney --risk-pct 1 --min-ml-confidence 0.55 --exit-mode fixed
```

Normal modun başabaş ve pozitif geçmiş kanıtı şartları devam eder. `--deney`, önceki kullanıcı tercihiyle bu iki şartı atlayarak demo ileri testi yapar; %65 hedef ve %55 ham yön kontrollerini atlamaz. %55 yön güveni, kalibre edilmiş işlem kazanma olasılığı değildir.

Veritabanına işlem başına `exit_mode` eklenir. Eski kayıtlar `trailing` olarak taşınır; açık eski pozisyonların lotu ve çıkış kuralı yeni varsayılan yüzünden değiştirilmez. Yeni işlemler `fixed` olarak kaydedilir. Broker TP kapanışları günlükte `HEDEF` olarak görünür. İstenirse yeni işlemlerde eski takip modu `--exit-mode trailing` ile seçilebilir.

[Araştırma raporundaki](REPORT.md) seçenekler tek tek incelendi. %55 yön filtresi, %1 risk ve sabit hedefin birleşimi yeni bir ileri demo düzenidir; bu birleşimin pozitif beklenen sonucu doğrulanmış değildir. Araştırma raporu uygulama öncesi snapshot olarak korunmuştur.
