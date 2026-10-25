// Public BID candles for prospective research; writes a separate 60-day archive.
const { getHistoricalRates } = require('dukascopy-node');
const fs = require('fs');
const path = require('path');
const pairs = ['eurusd', 'gbpusd', 'usdjpy', 'audusd', 'usdcad', 'usdchf', 'nzdusd', 'eurchf'];
const root = path.resolve(__dirname, '..', 'data', 'forward_prices');
const statePath = path.join(root, 'status.json');
const watchIndex = process.argv.indexOf('--watch');
const interval = watchIndex >= 0 ? Number(process.argv[watchIndex + 1]) : 0;
if (!Number.isFinite(interval) || (interval && interval < 60)) throw Error('Watch seconds must be >= 60');

async function update() {
  fs.mkdirSync(root, { recursive: true });
  const now = new Date();
  const to = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()));
  const from = new Date(to.getTime() - 60 * 86400000);
  let previous = {};
  if (fs.existsSync(statePath)) previous = JSON.parse(fs.readFileSync(statePath, 'utf8'));
  const status = { checked_at: now.toISOString(), closed_day_end: to.toISOString(), source: 'Dukascopy public BID candles', pairs: {}, errors: [] };
  for (const pair of pairs) {
    for (const [tf, folder] of [['m5', 'historical_5m'], ['m15', 'historical_15m']]) {
      const key = `${pair}/${tf}`;
      const directory = path.join(root, folder);
      const target = path.join(directory, `${pair.toUpperCase()}.csv`);
      if (!target.startsWith(root + path.sep)) throw Error('Target outside forward archive');
      if (previous.closed_day_end === to.toISOString() && previous.pairs?.[key] &&
          to.getTime() - previous.pairs[key].last_timestamp <= 4 * 86400000 && fs.existsSync(target)) {
        status.pairs[key] = previous.pairs[key];
        continue;
      }
      try {
        let csv = await getHistoricalRates({ instrument: pair, dates: { from, to }, timeframe: tf,
          priceType: 'bid', format: 'csv', useCache: true, cacheFolderPath: path.join(__dirname, 'cache'),
          batchSize: 10, pauseBetweenBatchesMs: 300, retries: 2, retryOnEmptyPacketRetryLimit: 1 });
        let lines = csv.trim().split(/\r?\n/);
        if (lines.length < 100 || !lines[0].startsWith('timestamp,')) throw Error('Empty or unexpected candles');
        if (to.getTime() - Number(lines.at(-1).split(',')[0]) > 4 * 86400000) {
          // Do not confuse a partial old cache with up-to-date provider history.
          const latest = await getHistoricalRates({ instrument: pair,
            dates: { from: new Date(to.getTime() - 2 * 86400000), to }, timeframe: tf,
            priceType: 'bid', format: 'csv', useCache: false, batchSize: 4, retries: 1 });
          const tail = typeof latest === 'string' ? latest.trim().split(/\r?\n/).slice(1).filter(Boolean) : [];
          if (tail.length) {
            const merged = new Map(lines.slice(1).map(line => [Number(line.split(',')[0]), line]));
            for (const line of tail) merged.set(Number(line.split(',')[0]), line);
            lines = [lines[0], ...[...merged].sort((a,b) => a[0]-b[0]).map(pair => pair[1])];
            csv = lines.join('\n') + '\n';
          }
        }
        fs.mkdirSync(directory, { recursive: true });
        fs.writeFileSync(target + '.tmp', csv, 'utf8');
        fs.renameSync(target + '.tmp', target);
        const last = Number(lines.at(-1).split(',')[0]);
        status.pairs[key] = { rows: lines.length - 1, last_timestamp: last,
          stale: to.getTime() - last > 4 * 86400000 };
        console.log(`${key}: ${lines.length - 1} closed-history candles`);
      } catch (error) {
        status.errors.push({ key, error: error.message });
        console.error(`${key}: download failed`);
      }
    }
  }
  status.stale_archives = Object.values(status.pairs).filter(record => record.stale).length;
  status.availability = status.errors.length || !Object.keys(status.pairs).length ? 'download_incomplete' :
    status.stale_archives ? 'partial_or_stale_provider_history' : 'recent_closed_history';
  fs.writeFileSync(statePath + '.tmp', JSON.stringify(status, null, 2), 'utf8');
  fs.renameSync(statePath + '.tmp', statePath);
  console.log(JSON.stringify({ checked_at: status.checked_at, errors: status.errors.length, archives: Object.keys(status.pairs).length }));
  if (status.errors.length && !interval) process.exitCode = 1;
}

async function main() {
  do {
    await update();
    if (interval) await new Promise(resolve => setTimeout(resolve, interval * 1000));
  } while (interval);
}
main().catch(() => { console.error('Forward archive updater failed'); process.exitCode = 1; });
