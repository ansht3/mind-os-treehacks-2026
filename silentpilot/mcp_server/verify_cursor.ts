
import { browserGoto, browserType, browserScreenshot, closeBrowser } from "./src/tools/index";
import * as fs from "fs";

async function main() {
    try {
        console.log("Navigating to google.com...");
        await browserGoto("https://www.google.com");

        console.log("Typing 'hello world'...");
        // Wait a bit to ensure page is loaded
        await new Promise(r => setTimeout(r, 2000));

        // This will trigger cursor movement and type. Selector for Google search box.
        // It's usually a textarea with name='q' or title='Search'.
        // To be safe, try textarea[name='q']
        await browserType("textarea[name='q']", "hello world");

        console.log("Taking screenshot...");
        const b64 = await browserScreenshot();
        fs.writeFileSync("cursor_verify.png", Buffer.from(b64, 'base64'));
        console.log("Screenshot saved to cursor_verify.png");
    } catch (e) {
        console.error("Error:", e);
    } finally {
        await closeBrowser();
    }
}

main();
