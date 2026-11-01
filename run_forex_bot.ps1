# forex_bot.py'yi hatayla durursa yeniden baslatir; Ctrl+C botu gercekten durdurur.
# Her acilista demo hesap kontrolu yeniden calisir. Cikti data\bot_run.log dosyasina da yazilir.
#   powershell -ExecutionPolicy Bypass -File run_forex_bot.ps1 --deney --risk-pct 1 --exit-mode fixed
$log = Join-Path $PSScriptRoot "data\bot_run.log"
$bot = Join-Path $PSScriptRoot "forex_bot.py"
$retrySeconds = 70  # heartbeat suresi (60 sn) dolmadan acilan bot "baska bot calisiyor" der
[Console]::OutputEncoding = [Text.Encoding]::UTF8

function Write-Run($line) {
    "$line"
    Add-Content -Path $log -Value "$line" -Encoding utf8
}

while ($true) {
    & python -X utf8 -u $bot @args 2>&1 | ForEach-Object { Write-Run $_ }
    $code = $LASTEXITCODE
    if ($code -eq 0) { break }
    Write-Run "$(Get-Date -Format 'dd.MM HH:mm:ss')  Bot hatayla durdu (kod $code); $retrySeconds sn sonra yeniden baslatiliyor."
    Start-Sleep -Seconds $retrySeconds
}
