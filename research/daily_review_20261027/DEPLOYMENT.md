# %10 maliyet sınırının demo botuna uygulanması

27 Ekim 2026: Kullanıcı günlük incelemeden sonra düzenlemelerin commit edilip botun yeniden başlatılmasını istedi.

```powershell
python -X utf8 -u forex_bot.py --deney --risk-pct 1 --min-ml-confidence 0.55 --exit-mode fixed --max-cost-share 0.10
```

Yeni girişlerde toplam tahmini maliyetin (spread + komisyon) toplam planlanan kayba oranı, `maliyet / (stop + maliyet)`, en fazla %10 olabilir. Tam %10 kabul edilir; üzeri engellenir. Bu kontrol `--deney` kipinde de uygulanır. Hedef kontrolü %65, ham yön güveni %55, risk bütçesi %1 ve sabit TP korunur. Haber, anlık spread, pozisyon, para birimi ve teminat kontrolleri devam eder.

Bu bir yeni giriş filtresidir; açık pozisyonları zorla kapatmaz, lotlarını veya kayıtlı çıkış yöntemlerini değiştirmez. Diğer panel ve plan kullanıcıları için `trading_blockers` varsayılanı %15 kalır; bot kendi %10 sınırını açıkça geçirir.

[Günlük rapor](REPORT.md) ve `cost_cap_history.json` uygulama öncesi araştırmadır. Daha düşük maliyet sınırı tarihsel kaybı azalttı; pozitif beklenen sonuç veya zarar etmeme kanıtı değildir. Yeni kurgu demo ileri test olarak çalışır.
