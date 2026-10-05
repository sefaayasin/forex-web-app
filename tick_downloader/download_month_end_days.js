// Validation data for research/month_end/validation_protocol.json: Dukascopy m5 bid bars for the last
// three business days of each month, 2003-06 .. 2007-12, for the 7 USD pairs. Output: research/month_end/m5_2003_2007/<PAIR>.csv
const { getHistoricalRates } = require('dukascopy-node');
const fs = require('fs');
const path = require('path');

const PAIRS = ['eurusd', 'gbpusd', 'audusd', 'nzdusd', 'usdjpy', 'usdcad', 'usdchf'];
const OUT_DIR = path.join(__dirname, '..', 'research', 'month_end', 'm5_2003_2007');
const CACHE_DIR = path.join(__dirname, 'cache');

function lastBusinessDays(year, month, count) {
  const days = [];
  for (let d = new Date(Date.UTC(year, month + 1, 0)); days.length < count; d.setUTCDate(d.getUTCDate() - 1)) {
    const wd = d.getUTCDay();
    if (wd !== 0 && wd !== 6) days.push(new Date(d));
  }
  return days;
}

async function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  for (const pair of PAIRS) {
    const outPath = path.join(OUT_DIR, `${pair.toUpperCase()}.csv`);
    if (fs.existsSync(outPath)) { console.log(`[SKIP] ${pair}`); continue; }
    const rows = [];
    for (let year = 2003; year <= 2007; year++) {
      for (let month = (year === 2003 ? 5 : 0); month < 12; month++) {
        for (const day of lastBusinessDays(year, month, 3)) {
          const from = new Date(Date.UTC(day.getUTCFullYear(), day.getUTCMonth(), day.getUTCDate(), 12));
          const to = new Date(Date.UTC(day.getUTCFullYear(), day.getUTCMonth(), day.getUTCDate(), 18));
          try {
            const data = await getHistoricalRates({ instrument: pair, dates: { from, to }, timeframe: 'm5', format: 'json',
              useCache: true, cacheFolderPath: CACHE_DIR, retryOnEmptyPacketRetryLimit: 2 });
            for (const r of data) rows.push(`${r.timestamp},${r.open},${r.high},${r.low},${r.close},${r.volume ?? 0}`);
          } catch (err) {
            console.error(`[HATA] ${pair} ${day.toISOString().slice(0, 10)}: ${err.message}`);
          }
        }
      }
    }
    fs.writeFileSync(outPath, 'timestamp,open,high,low,close,volume\n' + rows.join('\n') + '\n');
    console.log(`[OK] ${pair}: ${rows.length} bars`);
  }
}
main();
