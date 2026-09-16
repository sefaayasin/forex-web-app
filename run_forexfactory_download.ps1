$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$logFile = Join-Path $PSScriptRoot "download_forexfactory_calendar.log"
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
try {
    $output = & "C:\Users\Sefa\AppData\Local\Programs\Python\Python313\python.exe" download_forexfactory_calendar.py 2>&1
    Add-Content -Path $logFile -Value "[$timestamp] $output"
} catch {
    Add-Content -Path $logFile -Value "[$timestamp] HATA: $_"
}
