// Unseen 2003-2007 holdout for pre-registered research (research/night_bracket).
// Kept in a separate folder so no development script reads it by accident.
const { getHistoricalRates } = require('dukascopy-node');
const fs = require('fs');
const path = require('path');

const PAIRS = [
  'eurusd', 'gbpusd', 'usdjpy', 'audusd', 'usdcad', 'usdchf', 'nzdusd',
  'eurgbp', 'eurchf', 'eurjpy', 'gbpchf', 'gbpjpy', 'audjpy', 'euraud',
  'eurcad', 'audnzd', 'audcad', 'cadjpy', 'chfjpy', 'gbpaud', 'gbpcad',
];
const TIMEFRAMES = ['d1', 'h4', 'h1', 'm15', 'm5'];
const FROM = new Date('2003-01-01');
const TO = new Date('2008-01-01');
const OUT_ROOT = path.join(__dirname, '..', 'data', 'holdout_2003_2007');
const CACHE_DIR = path.join(__dirname, 'cache');
const DIR_LABELS = { m5: '5m', m15: '15m', h1: '1h', h4: '4h', d1: '1d' };

async function downloadOne(pair, timeframe) {
  const outDir = path.join(OUT_ROOT, `historical_${DIR_LABELS[timeframe]}`);
  fs.mkdirSync(outDir, { recursive: true });
  const outPath = path.join(outDir, `${pair.toUpperCase()}.csv`);
  if (fs.existsSync(outPath)) {
    console.log(`[SKIP] ${pair}/${timeframe}`);
    return;
  }
  const start = Date.now();
  try {
    const data = await getHistoricalRates({
      instrument: pair, dates: { from: FROM, to: TO }, timeframe, format: 'json',
      useCache: true, cacheFolderPath: CACHE_DIR, batchSize: 20,
      pauseBetweenBatchesMs: 1000, retryOnEmptyPacketRetryLimit: 2,
    });
    const rows = data.map((r) => `${r.timestamp},${r.open},${r.high},${r.low},${r.close},${r.volume ?? 0}`);
    fs.writeFileSync(outPath, 'timestamp,open,high,low,close,volume\n' + rows.join('\n') + '\n');
    console.log(`[OK] ${pair}/${timeframe}: ${rows.length} rows, ${((Date.now() - start) / 1000).toFixed(1)}s`);
  } catch (err) {
    console.error(`[ERR] ${pair}/${timeframe}: ${err.message}`);
  }
}

async function main() {
  for (const timeframe of TIMEFRAMES) {
    for (const pair of PAIRS) await downloadOne(pair, timeframe);
  }
  console.log('DONE');
}

main();
