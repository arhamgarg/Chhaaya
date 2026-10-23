const fs = require('node:fs');
// Opt-in offline E2E. Start browser_demo_server.py first. Use an existing
// Playwright module and Chromium executable, without adding a CI dependency.
const [modulePath, executablePath, artifactPath] = process.argv.slice(2);
if (!modulePath || !executablePath || !artifactPath) {
  throw new Error('Usage: node tests/browser_demo_retry.cjs PLAYWRIGHT_MODULE CHROMIUM OUTPUT_JSON');
}
const { chromium } = require(modulePath);
(async () => {
  const browser = await chromium.launch({executablePath});
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto('http://127.0.0.1:8127/?key=synthetic-demo-key');
    await page.locator('#running[data-state="running"]').waitFor();
    const result = await page.evaluate(fs.readFileSync('tests/browser_demo_retry.js','utf8'));
    const before = result.state;
    // Reload after a request commits while its response is still blocked.
    await page.evaluate(async () => {
      const realFetch = window.fetch.bind(window);
      window.fetch = async (url, options) => {
        const response = await realFetch(url, options);
        if (url === '/api/messages' && options?.method === 'POST') {
          window.testCommitted = true;
          await new Promise(() => {});
        }
        return response;
      };
      const box = document.getElementById('patient-text');
      box.value = 'namaste refresh during send';
      box.dispatchEvent(new Event('input'));
      document.getElementById('patient-form').requestSubmit();
    });
    await page.waitForFunction(() => window.testCommitted);
    await page.reload();
    await page.locator('#running[data-state="running"]').waitFor();
    await page.waitForFunction(async () => {
      const s = await (await fetch('/api/state')).json();
      return s.patient.some(m => m.body === 'namaste refresh during send' && m.status === 'done');
    });
    const after = await page.evaluate(async () => (await fetch('/api/state')).json());
    const historyPreserved = before.patient.every(m => after.patient.some(n => n.id === m.id)) && before.cases.every(c => after.cases.some(n => n.number === c.number));
    result.results.push({name:'refresh during committed request preserves conversation and cases',status:historyPreserved?'PASS':'FAIL'});
    result.pageErrors = errors;
    result.passed = result.results.every(r => r.status === 'PASS') && errors.length === 0;
    result.command = process.argv.join(' ');
    fs.writeFileSync(artifactPath,JSON.stringify(result,null,2)+'\n');
    console.log(JSON.stringify({passed:result.passed,results:result.results,pageErrors:errors},null,2));
    process.exitCode = result.passed ? 0 : 1;
  } finally { await browser.close(); }
})().catch(error => {console.error(error);process.exitCode=1;});
