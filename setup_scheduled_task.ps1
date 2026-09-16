$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument '-NoProfile -ExecutionPolicy Bypass -File "C:\Users\Sefa\Desktop\Yapay Zeka Projeleri\yeni_forex\run_forexfactory_download.ps1"'
$trigger = New-ScheduledTaskTrigger -Daily -At 7:00AM
Register-ScheduledTask -TaskName "ForexFactoryCalendarDownload" -Action $action -Trigger $trigger -Description "Ekonomik haber takvimini gunluk olarak indirir (forex_analiz projesi)" -Force
Get-ScheduledTask -TaskName "ForexFactoryCalendarDownload" | Select-Object TaskName, State
