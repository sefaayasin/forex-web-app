const { getHistoricalRates } = require('dukascopy-node');
const fs = require('fs');
const path = require('path');

const PAIRS = [
  'eurusd', 'gbpusd', 'usdjpy', 'audusd', 'usdcad', 'usdchf', 'nzdusd',
  'eurgbp', 'euraud', 'eurcad', 'eurchf', 'eurjpy', 'eurnzd',
  'gbpjpy', 'gbpaud', 'gbpcad', 'gbpchf', 'gbpnzd',
  'audcad', 'audchf', 'audjpy', 'audnzd',
  'cadchf', 'cadjpy', 'chfjpy',
  'nzdcad', 'nzdchf', 'nzdjpy', 'eurzar',
];

// Dukascopy'nin native olarak sunduğu zaman dilimleri (w1 yok, çünkü hiçbir
// veri sağlayıcı haftalık mumu ham olarak üretmez; d1'den türetilecek).
const TIMEFRAMES = ['m5', 'm15', 'm30', 'h1', 'h4', 'd1'];

const FROM = new Date('2008-01-01');
const TO = new Date();

const OUT_ROOT = path.join(__dirname, '..', 'data');
const CACHE_DIR = path.join(__dirname, 'cache');

const DIR_LABELS = { m5: '5m', m15: '15m', m30: '30m', h1: '1h', h4: '4h', d1: '1d' };

async function downloadOne(pair, timeframe) {
  const outDir = path.join(OUT_ROOT, `historical_${DIR_LABELS[timeframe]}`);
  fs.mkdirSync(outDir, { recursive: true });
  const outPath = path.join(outDir, `${pair.toUpperCase()}.csv`);

  if (fs.existsSync(outPath)) {
    console.log(`[SKIP] ${pair}/${timeframe}: zaten var`);
    return;
  }

  const start = Date.now();
  try {
    const csv = await getHistoricalRates({
      instrument: pair,
      dates: { from: FROM, to: TO },
      timeframe,
      format: 'csv',
      useCache: true,
      cacheFolderPath: CACHE_DIR,
      batchSize: 20,
      pauseBetweenBatchesMs: 1000,
      retryOnEmptyPacketRetryLimit: 2,
    });
    fs.writeFileSync(outPath, csv);
    const secs = ((Date.now() - start) / 1000).toFixed(1);
    const lines = csv.split('\n').length - 1;
    console.log(`[OK] ${pair}/${timeframe}: ${lines} satır, ${secs}s`);
  } catch (err) {
    console.error(`[HATA] ${pair}/${timeframe}: ${err.message}`);
  }
}

async function main() {
  for (const timeframe of TIMEFRAMES) {
    console.log(`\n=== ZAMAN DİLİMİ: ${timeframe} ===`);
    for (const pair of PAIRS) {
      await downloadOne(pair, timeframe);
    }
  }
  console.log('\nTÜM ZAMAN DİLİMLERİ TAMAMLANDI.');
}

main();
