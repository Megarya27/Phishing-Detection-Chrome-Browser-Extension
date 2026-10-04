//This script is hte background service worker
// content scripts send scraped email/link data here, this forwards it to  flask api
//keeps track of latest result per tab so popup can read back

const DEFAULT_API_BASE_URL = "http://localhost:5000";
const DECISION_THRESHOLD = 0.1; // this matche hybrid_report.decision_threshold in artifacts/results/hybrid_fusion_report.json
const latestResultsByTab = new Map(); // tabId -> {emailText, links, scanFailed, scannedAt}
const activeScanningTabs = new Set(); // tabId currently scanning so popup shows spinner
const LATENCY_LOG_STORAGE_KEY = "latencyLog";
const MAX_LATENCY_LOG_ENTRIES = 5000; // limits log entries

//appends new latency measurements to local storage, keeps only most recent entries
async function appendLatencyRecords(newRecords) {
  const stored = await chrome.storage.local.get([LATENCY_LOG_STORAGE_KEY]);
  const existing = stored[LATENCY_LOG_STORAGE_KEY] || [];
  const combined = existing.concat(newRecords).slice(-MAX_LATENCY_LOG_ENTRIES);
  await chrome.storage.local.set({ [LATENCY_LOG_STORAGE_KEY]: combined });}

//get user settings (like a custom api url) from local storage, falls back to defaults
async function getExtensionSettings() {
  const stored = await chrome.storage.local.get(["apiBaseUrl"]);
  return {
    apiBaseUrl: stored.apiBaseUrl || DEFAULT_API_BASE_URL,
    decisionThreshold: DECISION_THRESHOLD};}

//update extension icon red badge with number of flagged links in current tab
function updateTabBadge(tabId, flaggedLinkCount) {
  if (flaggedLinkCount > 0) {
    chrome.action.setBadgeText({ tabId, text: String(flaggedLinkCount) });
    chrome.action.setBadgeBackgroundColor({ tabId, color: "#d32f2f" });} 
  else {
    chrome.action.setBadgeText({ tabId, text: "" });}}

//handles scan state changes, settings queries, log management and api requests
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "SCAN_STARTED") {
    const tabId = sender.tab && sender.tab.id;
    if (tabId != null) {
      activeScanningTabs.add(tabId);
      console.log("[Phishing Guard][background] scan started for tab", tabId);
      chrome.runtime.sendMessage({ type: "SCAN_STATUS_CHANGED", tabId }).catch(() => { });}
    return false;}

  if (message.type === "SCAN_RESULT") {
    const tabId = sender.tab && sender.tab.id;
    console.log("[Phishing Guard][background] scan result for tab", tabId, "-", message.links.length, "link(s)");
    if (tabId != null) {
      latestResultsByTab.set(tabId, {
        emailText: message.emailText,
        links: message.links,
        scanFailed: Boolean(message.scanFailed),
        scannedAt: Date.now()});
      activeScanningTabs.delete(tabId);
      const flaggedLinkCount = message.links.filter((link) => link.warning_shown).length;
      updateTabBadge(tabId, flaggedLinkCount);
      chrome.runtime.sendMessage({ type: "SCAN_STATUS_CHANGED", tabId }).catch(() => { });}
    return false;}

  if (message.type === "GET_THRESHOLD") {
    sendResponse({ threshold: DECISION_THRESHOLD });
    return false;}

  if (message.type === "LOG_LATENCY_RESULTS") {
    appendLatencyRecords(message.records || []);
    return false;}

  if (message.type === "EXPORT_LATENCY_LOG") {
    (async () => {
      const stored = await chrome.storage.local.get([LATENCY_LOG_STORAGE_KEY]);
      const records = stored[LATENCY_LOG_STORAGE_KEY] || [];

      if (records.length === 0) {
        sendResponse({ error: "No latency records collected yet, scan some emails first." });
        return;}

      // data: URL instead of a Blob URL - service workers can't hold onto object URLs
      const jsonText = JSON.stringify(records, null, 2);
      const dataUrl = "data:application/json;charset=utf-8," + encodeURIComponent(jsonText);

      chrome.downloads.download(
        { url: dataUrl, filename: `phishing_guard_latency_log_${Date.now()}.json`, saveAs: true },
        () => {
          if (chrome.runtime.lastError) {
            sendResponse({ error: chrome.runtime.lastError.message });
          } else {
            sendResponse({ n_records: records.length });
          }});
    })();
    return true;} //async sendResponse through downloads callback
  

  if (message.type === "CLEAR_LATENCY_LOG") {
    chrome.storage.local.set({ [LATENCY_LOG_STORAGE_KEY]: [] }, () => sendResponse({ ok: true }));
    return true;}

  if (message.type === "GET_SCAN_RESULT") {
    const tabData = latestResultsByTab.get(message.tabId) || null;
    const isScanning = activeScanningTabs.has(message.tabId);
    sendResponse({ scanning: isScanning, data: tabData });
    return false;}

  if (message.type !== "PHISHING_CHECK_REQUEST") {
    return false;}

  // fetch from here, not content script, so page content security policies cant block  request
  (async () => {
    const { apiBaseUrl, decisionThreshold } = await getExtensionSettings();
    const requestStart = performance.now();

    try {
      const apiResponse = await fetch(`${apiBaseUrl}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          url: message.payload.url,
          text: message.payload.text,
          structural_features: message.payload.structural_features,
        }),
      });

      if (!apiResponse.ok) {
        throw new Error(`Inference API returned HTTP ${apiResponse.status}`);}

      const prediction = await apiResponse.json();
      const apiRoundTripMs = performance.now() - requestStart;

      sendResponse({
        phishing_probability: prediction.phishing_probability,
        semantic_probability: prediction.semantic_probability,
        structural_probability: prediction.structural_probability,
        threshold: decisionThreshold,
        timings: {...message.client_timings,
          api_round_trip_ms: apiRoundTripMs,
          backend_processing_ms: prediction.backend_processing_ms,
        },
      });} 
    catch (err) {
      sendResponse({ error: err.message || String(err) });}
  })();
  return true; });// keep message port open for async sendResponse above

//clean up state when tab is closed to prevent memory leaks
chrome.tabs.onRemoved.addListener((closedTabId) => {
  latestResultsByTab.delete(closedTabId);
  activeScanningTabs.delete(closedTabId);});
