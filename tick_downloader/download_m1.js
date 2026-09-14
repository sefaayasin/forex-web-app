const { getHistoricalRates } = require('dukascopy-node');
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

const PAIRS = [
  'eurusd', 'gbpusd', 'usdjpy', 'audusd', 'usdcad', 'usdchf', 'nzdusd',
  'eurgbp', 'euraud', 'eurcad', 'eurchf', 'eurjpy', 'eurnzd',
  'gbpjpy', 'gbpaud', 'gbpcad', 'gbpchf', 'gbpnzd',
  'audcad', 'audchf', 'audjpy', 'audnzd',
  'cadchf', 'cadjpy', 'chfjpy',
  'nzdcad', 'nzdchf', 'nzdjpy', 'eurzar',
];

const FROM = new Date('2008-01-01');
const TO = new Date();

const OUT_DIR = path.join(__dirname, '..', 'data', 'historical_m1');
const CACHE_DIR = path.join(__dirname, 'cache');

async function downloadPair(pair) {
  const outPath = path.join(OUT_DIR, `${pair.toUpperCase()}.csv`);
  if (fs.existsSync(outPath)) {
    console.log(`[SKIP] ${pair}: zaten var (${outPath})`);
    return;
  }
  console.log(`[BAŞLADI] ${pair} ...`);
  const start = Date.now();
  try {
    const csv = await getHistoricalRates({
      instrument: pair,
      dates: { from: FROM, to: TO },
      timeframe: 'm1',
      format: 'csv',
      useCache: true,
      cacheFolderPath: CACHE_DIR,
      batchSize: 20,
      pauseBetweenBatchesMs: 1500,
      retryOnEmptyPacketRetryLimit: 2,
    });
    fs.writeFileSync(outPath, csv);
    const secs = ((Date.now() - start) / 1000).toFixed(1);
    console.log(`[OK] ${pair}: yazıldı, ${secs}s`);
  } catch (err) {
    console.error(`[HATA] ${pair}: ${err.message}`);
  }
}

async function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  fs.mkdirSync(CACHE_DIR, { recursive: true });
  for (const pair of PAIRS) {
    await downloadPair(pair);
  }
  console.log('\nTÜM İNDİRMELER TAMAMLANDI.');
}

main();
