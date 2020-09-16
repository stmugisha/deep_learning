const puppeteer = require('puppeteer');

async function run() {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();

    await page.goto('https://github.com/steph-en-m');
    await page.screenshot({path: 'screenshots/github.png'});

    browser.close();
}

run();