// Downloads investing.com economic-calendar history, one JSON file per week.
//
// Plain HTTP requests to investing.com are blocked (403), but the calendar's own
// AJAX endpoint answers from inside a real browser session, so this runs a
// headless Chromium via Playwright and calls the endpoint with fetch().
//
// Setup (once): npm install playwright   (reuses an installed Chromium if present)
// Run:          node research/calendar_sources/fetch_investing.js 2024-09-30 2026-09-28
// Output:       research/calendar_sources/raw/investing/<monday>.json  {"pages": [html, ...]}
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

// US, Euro Zone, Germany, France, Italy, Spain, UK, Japan, Canada, Australia, New Zealand, Switzerland
const COUNTRIES = ['5', '72', '17', '22', '10', '26', '4', '35', '6', '25', '43', '12'];
// Medium and high importance only: one request per week instead of two, because
// investing.com rate-limits (HTTP 429) hard. The first 7 weeks were fetched unfiltered.
const IMPORTANCE = ['2', '3'];
const OUT_DIR = path.join(__dirname, 'raw', 'investing');

function mondays(start, end) {
  const out = [];
  for (let d = new Date(start + 'T00:00:00Z'); d <= new Date(end + 'T00:00:00Z'); d.setUTCDate(d.getUTCDate() + 7)) {
    out.push(d.toISOString().slice(0, 10));
  }
  return out;
}

async function fetchWeek(page, monday) {
  const sunday = new Date(monday + 'T00:00:00Z');
  sunday.setUTCDate(sunday.getUTCDate() + 6);
  return page.evaluate(async ({ from, to, countries, IMPORTANCE }) => {
    const pages = [];
    let limitFrom = 0;
    let lastScope = null;
    for (;;) {
      const body = new URLSearchParams({ dateFrom: from, dateTo: to, timeZone: '55', timeFilter: 'timeOnly', currentTab: 'custom', limit_from: String(limitFrom) });
      countries.forEach(c => body.append('country[]', c));
      IMPORTANCE.forEach(i => body.append('importance[]', i));
      if (lastScope) body.append('last_time_scope', String(lastScope));
      const r = await fetch('/economic-calendar/Service/getCalendarFilteredData', {
        method: 'POST', body, headers: { 'X-Requested-With': 'XMLHttpRequest', 'Content-Type': 'application/x-www-form-urlencoded' },
      });
      if (r.status !== 200) throw new Error('HTTP ' + r.status);
      const data = await r.json();
      pages.push(data.data);
      if (!data.bind_scroll_handler || limitFrom > 20) break;
      limitFrom += 1;
      lastScope = data.last_time_scope;
    }
    return pages;
  }, { from: monday, to: sunday.toISOString().slice(0, 10), countries: COUNTRIES, IMPORTANCE });
}

(async () => {
  const [start, end] = process.argv.slice(2);
  fs.mkdirSync(OUT_DIR, { recursive: true });
  const browser = await chromium.launch();
  const page = await browser.newPage({ userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36' });
  // Only the origin matters (the AJAX call must be same-origin); the page itself may hang on a challenge.
  await page.goto('https://www.investing.com/economic-calendar/', { timeout: 60000, waitUntil: 'commit' }).catch(e => console.log('goto:', e.message));
  await page.waitForTimeout(5000);
  // investing.com answers HTTP 429 after a few quick requests; go slowly and back off.
  for (const monday of mondays(start, end)) {
    const file = path.join(OUT_DIR, monday + '.json');
    if (fs.existsSync(file)) continue;
    let pages = null;
    for (let attempt = 0; attempt < 6 && !pages; attempt++) {
      try {
        pages = await fetchWeek(page, monday);
      } catch (e) {
        console.log(monday, 'retry after', e.message);
        await page.waitForTimeout(60000 * (attempt + 1));
      }
    }
    if (!pages) throw new Error('gave up on ' + monday);
    fs.writeFileSync(file, JSON.stringify({ pages }));
    console.log(monday, pages.length, 'page(s)');
    await page.waitForTimeout(30000);
  }
  await browser.close();
})().catch(e => { console.error('ERR', e.message); process.exit(1); });
