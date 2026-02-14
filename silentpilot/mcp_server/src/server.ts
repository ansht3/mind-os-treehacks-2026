/**
 * MCP Server with SSE transport for SilentPilot.
 *
 * Exposes 8 Playwright-backed browser tools via the Model Context Protocol.
 * The OpenAI Responses API connects to this server to control a browser.
 *
 * Transport: SSE (Server-Sent Events) over Express
 * - GET /sse -> SSE connection endpoint
 * - POST /messages -> message handler
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";
import express from "express";
import { z } from "zod";

import {
  browserGoto,
  browserClick,
  browserType,
  browserPress,
  browserScroll,
  browserScreenshot,
  browserExtractText,
  browserWait,
  closeBrowser,
} from "./tools/index.js";

const PORT = parseInt(process.env.MCP_PORT || "3333", 10);

// --- Create MCP Server ---

const server = new McpServer({
  name: "silentpilot-browser",
  version: "0.1.0",
});

// --- Register Tools ---

server.tool(
  "browser_goto",
  "Navigate the browser to a URL",
  { url: z.string().describe("The URL to navigate to") },
  async ({ url }) => ({
    content: [{ type: "text", text: await browserGoto(url) }],
  })
);

server.tool(
  "browser_click",
  "Click an element on the page by CSS selector",
  { selector: z.string().describe("CSS selector of element to click") },
  async ({ selector }) => ({
    content: [{ type: "text", text: await browserClick(selector) }],
  })
);

server.tool(
  "browser_type",
  "Type text into an input element",
  {
    selector: z.string().describe("CSS selector of input element"),
    text: z.string().describe("Text to type"),
  },
  async ({ selector, text }) => ({
    content: [{ type: "text", text: await browserType(selector, text) }],
  })
);

server.tool(
  "browser_press",
  "Press a keyboard key (e.g., Enter, Tab, Escape)",
  { key: z.string().describe("Key to press (e.g., 'Enter', 'Tab')") },
  async ({ key }) => ({
    content: [{ type: "text", text: await browserPress(key) }],
  })
);

server.tool(
  "browser_scroll",
  "Scroll the page up or down",
  {
    direction: z.enum(["up", "down"]).describe("Scroll direction"),
    amount: z.number().optional().default(300).describe("Pixels to scroll"),
  },
  async ({ direction, amount }) => ({
    content: [{ type: "text", text: await browserScroll(direction, amount) }],
  })
);

server.tool(
  "browser_screenshot",
  "Take a screenshot of the current page (returns base64 PNG)",
  {},
  async () => {
    const base64 = await browserScreenshot();
    return {
      content: [
        {
          type: "image",
          data: base64,
          mimeType: "image/png",
        },
      ],
    };
  }
);

server.tool(
  "browser_extract_text",
  "Extract visible text content from the current page",
  {},
  async () => ({
    content: [{ type: "text", text: await browserExtractText() }],
  })
);

server.tool(
  "browser_wait",
  "Wait for a specified duration in milliseconds",
  { ms: z.number().describe("Milliseconds to wait (max 10000)") },
  async ({ ms }) => ({
    content: [{ type: "text", text: await browserWait(ms) }],
  })
);

// --- Express + SSE Transport ---

const app = express();

// Store transports by session
const transports: Record<string, SSEServerTransport> = {};

app.get("/sse", async (req, res) => {
  console.log("[MCP] New SSE connection");
  const transport = new SSEServerTransport("/messages", res);
  transports[transport.sessionId] = transport;
  
  res.on("close", () => {
    console.log(`[MCP] SSE connection closed: ${transport.sessionId}`);
    delete transports[transport.sessionId];
  });
  
  await server.connect(transport);
});

app.post("/messages", async (req, res) => {
  const sessionId = req.query.sessionId as string;
  const transport = transports[sessionId];
  
  if (!transport) {
    res.status(400).json({ error: "Unknown session" });
    return;
  }
  
  await transport.handlePostMessage(req, res);
});

// --- Health check ---
app.get("/health", (_req, res) => {
  res.json({ status: "ok", tools: 8 });
});

// --- Start ---

app.listen(PORT, () => {
  console.log(`[MCP Server] Listening on http://localhost:${PORT}`);
  console.log(`[MCP Server] SSE endpoint: http://localhost:${PORT}/sse`);
  console.log(`[MCP Server] For OpenAI: expose via ngrok http ${PORT}`);
});

// Cleanup on exit
process.on("SIGINT", async () => {
  console.log("\n[MCP Server] Shutting down...");
  await closeBrowser();
  process.exit(0);
});
