const puppeteer = require('puppeteer');

async function run() {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();

    await page.goto('https://kaggle.com/competitions');
    await page.screenshot({path: 'screenshots/kaggle.png'});

    browser.close();
}

run();