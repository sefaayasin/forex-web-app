const { getRealTimeRates } = require('dukascopy-node');
getRealTimeRates({ instrument: 'eurusd', timeframe: 'm15', format: 'json', last: 10, priceType: 'bid' })
  .then(rows => console.log(JSON.stringify({ rows: rows.length, last_at: rows.length ? new Date(rows.at(-1).timestamp).toISOString() : null })))
  .catch(error => { console.error(error.name); process.exitCode = 1; });
