"""
local_llm_client/web/web_ui.py — Embedded Web UI for Local LLM Cloud  v1.0.0

Modern single‑page application with streaming chat, conversation persistence,
model management, syntax highlighting, robust retry/resume capabilities,
and a fully accessible, responsive interface.
"""

HTML = r"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Local LLM Cloud</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>⚡</text></svg>">
<style>
/* ═══════════════════════════════════════ RESET + BASE */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { height: 100%; font-size: 15px; }
:root {
  /* Content column max width. 94vw means on any window narrower than
     ~1490px the column fills 94% of the viewport; on wider monitors it
     caps at 1400px to keep lines readable. Pure CSS — no JS, so it
     resizes with the window automatically. */
  --content-max-w: min(1400px, 94vw);
}
body {
       height: 100vh; overflow: hidden;
       font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif;
       background: #0d0d0d; color: #e8e8e8;
       display: flex;
}
/* ═══════════════════════════════════════ SCROLLBAR */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #383838; }
::-webkit-scrollbar-track { background: transparent; }
/* ═══════════════════════════════════════ LIGHT THEME OVERRIDE */
[data-theme="light"] {
       --bg-main: #fafafa; --bg-sidebar: #fff; --bg-chat: #fff; --bg-bubble: #f0f0f0;
       --text-primary: #1a1a1a; --text-secondary: #555; --border-color: #ddd;
       --accent: #7c3aed;
}
/* ═══════════════════════════════════════ SIDEBAR */
#sidebar {
       width: 260px; min-width: 260px;
       background: #111; border-right: 1px solid #1e1e1e;
       display: flex; flex-direction: column;
       transition: width .22s ease, min-width .22s ease;
       overflow: hidden;
}
#sidebar.closed { width: 0; min-width: 0; border-right: none; }
.sb-head {
       padding: 14px 12px 10px; border-bottom: 1px solid #1e1e1e;
       display: flex; align-items: center; gap: 8px; flex-shrink: 0;
}
.sb-logo { font-weight: 700; font-size: 13px; color: #a78bfa; flex: 1; white-space: nowrap; overflow: hidden; }
#btn-new {
       background: #1e1e1e; border: 1px solid #2e2e2e; color: #ccc;
       border-radius: 7px; padding: 5px 11px; font-size: 12px;
       cursor: pointer; white-space: nowrap; transition: all .15s;
       display: flex; align-items: center; gap: 5px;
}
#btn-new:hover { background: #2a2a2a; color: #fff; border-color: #3e3e3e; }
.sb-search {
       padding: 8px 12px; border-bottom: 1px solid #1a1a1a; flex-shrink: 0;
}
#hist-search {
       width: 100%; background: #161616; border: 1px solid #222;
       color: #e8e8e8; border-radius: 6px; padding: 6px 10px;
       font-size: 12px; font-family: inherit; outline: none;
}
#hist-search:focus { border-color: #7c3aed; }
#hist-list { flex: 1; overflow-y: auto; padding: 4px 0; }
.hi {
       display: flex; align-items: center; gap: 6px;
       padding: 7px 12px; font-size: 13px; color: #888;
       cursor: pointer; border-radius: 6px; margin: 1px 6px;
       transition: all .12s; white-space: nowrap; overflow: hidden;
       user-select: none;
}
.hi:hover { background: #1c1c1c; color: #e8e8e8; }
.hi.active { background: #1e1535; color: #c4b5fd; }
.hi-title { flex: 1; overflow: hidden; text-overflow: ellipsis; font-size: 12px; }
.hi-time { font-size: 10px; color: #444; flex-shrink: 0; }
.hi-del { opacity: 0; flex-shrink: 0; padding: 2px 5px; border-radius: 4px;
                font-size: 13px; color: #555; transition: all .12s; }
.hi:hover .hi-del { opacity: 1; }
.hi-del:hover { color: #f87171 !important; background: #2a1515; }
.sb-foot {
       padding: 10px 12px; border-top: 1px solid #1a1a1a;
       flex-shrink: 0;
}
.sb-stat { font-size: 11px; color: #444; line-height: 1.9; }
.sb-stat .sv { color: #666; }
/* ═══════════════════════════════════════ MAIN */
#main { flex: 1; display: flex; flex-direction: column; overflow: hidden; min-width: 0; }
/* ═══════════════════════════════════════ TOPBAR */
#topbar {
       height: 54px; background: #111; border-bottom: 1px solid #1e1e1e;
       display: flex; align-items: center; gap: 10px; padding: 0 14px;
       flex-shrink: 0; z-index: 10;
}
#btn-sb {
       background: none; border: none; color: #555; font-size: 19px;
       cursor: pointer; padding: 5px 7px; border-radius: 7px; line-height: 1;
       transition: all .12s;
}
#btn-sb:hover { background: #1e1e1e; color: #e8e8e8; }
#tb-title { font-weight: 600; font-size: 14px; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
#model-sel {
       background: #171717; border: 1px solid #2a2a2a; color: #e8e8e8;
       border-radius: 8px; padding: 6px 10px; font-size: 12px; font-family: inherit;
       cursor: pointer; outline: none; max-width: 200px;
       transition: border-color .15s;
}
#model-sel:focus { border-color: #7c3aed; }
#model-sel option { background: #171717; }
#model-sel option.loading { color: #f59e0b; }
#model-sel option.error { color: #f87171; }
.status-badge {
       display: flex; align-items: center; gap: 6px; font-size: 12px; color: #666;
       background: #161616; border: 1px solid #222; border-radius: 20px; padding: 4px 10px;
       flex-shrink: 0; white-space: nowrap;
       cursor: pointer;
}
.status-badge:hover { background: #1e1e1e; border-color: #3e3e3e; }
.dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.dot.on  { background: #22c55e; }
.dot.off { background: #ef4444; }
.dot.busy { background: #f59e0b; animation: pulse 1s ease-in-out infinite; }
@keyframes pulse { 0%,100%{opacity:1; transform: scale(1)} 50%{opacity:.35; transform: scale(1.1);} }
#btn-sys {
       background: none; border: 1px solid #222; color: #555; border-radius: 7px;
       padding: 5px 9px; font-size: 12px; cursor: pointer; transition: all .12s;
       white-space: nowrap;
}
#btn-sys:hover, #btn-sys.on { background: #1e1535; color: #a78bfa; border-color: #4c1d95; }
/* timeout dropdown */
.timeout-select {
       background: #171717; border: 1px solid #2a2a2a; color: #e8e8e8;
       border-radius: 8px; padding: 5px 8px; font-size: 11px; cursor: pointer;
       margin-left: 8px;
}
/* settings button */
#btn-settings {
       background: none; border: 1px solid #222; color: #555; border-radius: 7px;
       padding: 5px 9px; font-size: 12px; cursor: pointer; transition: all .12s;
       white-space: nowrap;
}
#btn-settings:hover, #btn-settings.on { background: #1e1535; color: #a78bfa; border-color: #4c1d95; }
/* theme toggle button */
#btn-theme {
       background: none; border: 1px solid #222; color: #555; border-radius: 7px;
       padding: 5px 9px; font-size: 12px; cursor: pointer; transition: all .12s;
       white-space: nowrap;
       display: none;
}
@media(min-width:601px){#btn-theme{display:flex;align-items:center;gap:4px}}
#btn-theme:hover { background: #1e1535; color: #a78bfa; border-color: #4c1d95; }
/* OFFLINE BANNER */
#offline-bar {
       display: none;
       background: #2c1a1a;
       border-bottom: 1px solid #7f1a1a;
       color: #fca5a5;
       padding: 8px 16px;
       font-size: 12px;
       text-align: center;
       flex-shrink: 0;
}
#offline-bar.show { display: block; }
#offline-bar button {
       background: #222;
       border: 1px solid #444;
       color: #ccc;
       margin-left: 12px;
       padding: 2px 8px;
       border-radius: 4px;
       cursor: pointer;
}
#offline-bar button:hover { background: #333; color: #fff; }
/* ═══════════════════════════════════════ CHAT */
#chat { flex: 1; overflow-y: auto; padding: 20px 0; scroll-behavior: smooth; }
/* ═══════════════════════════════════════ EMPTY STATE */
#empty {
       height: 100%; display: flex; flex-direction: column;
       align-items: center; justify-content: center; gap: 18px;
       text-align: center; padding: 40px;
}
.emp-icon { font-size: 52px; opacity: .12; }
.emp-title { font-size: 22px; font-weight: 700; color: #e8e8e8; }
.emp-sub { font-size: 14px; color: #555; max-width: 400px; line-height: 1.65; }
.cards {
       display: grid; grid-template-columns: repeat(2, 1fr);
       gap: 10px; max-width: 560px; width: 100%;
}
.card {
       background: #161616; border: 1px solid #222; border-radius: 10px;
       padding: 14px 16px; text-align: left; cursor: pointer; font-size: 13px;
       color: #888; line-height: 1.45; transition: all .15s;
}
.card:hover { border-color: #7c3aed; background: #1a1530; color: #e8e8e8; }
.card-title { display: block; font-weight: 600; color: #e8e8e8; margin-bottom: 3px; }
/* ═══════════════════════════════════════ MESSAGES */
.row { max-width: var(--content-max-w); margin: 0 auto; padding: 3px 20px; animation: fadein .18s ease; }
@keyframes fadein { from{opacity:0;transform:translateY(5px)} to{opacity:1;transform:none} }
.row.user { display: flex; justify-content: flex-end; }
.row.assistant { display: flex; justify-content: flex-start; }
.bub {
       max-width: 92%; padding: 11px 18px; font-size: 15px; line-height: 1.68;
       border-radius: 16px; position: relative;
}
.bub.user {
       background: #2d1f63; color: #e8dcff;
       border-bottom-right-radius: 4px;
}
.bub.assistant {
       background: #161616; color: #e8e8e8;
       border: 1px solid #222; border-bottom-left-radius: 4px;
}
.bub.assistant.interrupted {
       border-color: #f59e0b;
       background: #1f1a0f;
}
/* markdown */
.bub.assistant strong { color: #c4b5fd; }
.bub.assistant em     { color: #93c5fd; font-style: italic; }
.bub.assistant a      { color: #818cf8; }
.bub.assistant h1,.bub.assistant h2,.bub.assistant h3,
.bub.assistant h4 { color: #e8e8e8; font-weight: 600; margin: 14px 0 5px; }
.bub.assistant h1 { font-size: 1.22em; border-bottom: 1px solid #222; padding-bottom: 6px; }
.bub.assistant h2 { font-size: 1.12em; }
.bub.assistant h3 { font-size: 1.03em; }
.bub.assistant ul,.bub.assistant ol { padding-left: 22px; margin: 6px 0; }
.bub.assistant li { margin: 3px 0; }
.bub.assistant blockquote {
       border-left: 3px solid #7c3aed; padding-left: 14px; color: #888; margin: 8px 0;
}
.bub.assistant hr { border: none; border-top: 1px solid #222; margin: 12px 0; }
.bub.assistant p { margin: 6px 0; }
.bub.assistant p:first-child { margin-top: 0; }
.bub.assistant p:last-child  { margin-bottom: 0; }
.bub.assistant table {
       border-collapse: collapse; width: 100%; font-size: 13px; margin: 10px 0;
}
.bub.assistant th { background: #1e1e1e; color: #c4b5fd; font-weight: 600; }
.bub.assistant th,.bub.assistant td {
       border: 1px solid #2a2a2a; padding: 6px 10px; text-align: left;
}
.bub.assistant tr:nth-child(even) td { background: #131313; }
/* inline code (short spans within prose) */
.bub.assistant .inline-code {
       background: #0e0e0e; border: 1px solid #2a2a2a; border-radius: 4px;
       padding: 1px 5px; font-family: 'SF Mono','Fira Code','Cascadia Code',monospace;
       font-size: 13px; color: #7dd3fc;
}
/* code blocks */
.bub.assistant pre.code-block {
       background: #0a0a0a; border: 1px solid #262626; border-radius: 10px;
       margin: 10px 0; overflow: hidden; position: relative;
       box-shadow: 0 1px 3px rgba(0,0,0,.3);
}
.pre-head {
       display: flex; align-items: center; justify-content: space-between;
       padding: 7px 8px 7px 14px; background: #151515; border-bottom: 1px solid #232323;
}
.pre-lang { font-size: 11px; color: #666; font-family: monospace; text-transform: uppercase; letter-spacing: .03em; }
.pre-actions { display: flex; gap: 4px; }
.pre-btn {
       background: none; border: 1px solid transparent; color: #666; border-radius: 6px;
       padding: 4px 9px; font-size: 11px; cursor: pointer; transition: all .12s;
       display: flex; align-items: center; gap: 5px; font-family: inherit;
}
.pre-btn:hover { background: #202020; color: #e8e8e8; border-color: #333; }
.pre-btn.ok { color: #22c55e; }
.bub.assistant code.hl-code {
       display: block; padding: 13px 15px; background: none; border: none;
       font-family: 'SF Mono','Fira Code','Cascadia Code',monospace;
       font-size: 13px; line-height: 1.6; color: #e2e8f0;
       overflow-x: auto; white-space: pre; tab-size: 4;
}
/* syntax-highlight tokens */
.tok-c { color: #6b7280; font-style: italic; }
.tok-s { color: #86efac; }
.tok-n { color: #7dd3fc; }
.tok-k { color: #c4b5fd; font-weight: 600; }
.tok-f { color: #93c5fd; }
/* cursor */
.cursor {
       display: inline-block; width: 2px; height: 1em; margin-left: 1px;
       background: #a78bfa; vertical-align: text-bottom;
       animation: blink .8s step-end infinite;
}
@keyframes blink { 50%{opacity:0} }
/* thinking */
.think-row { max-width: 800px; margin: 0 auto; padding: 6px 20px; }
.think-block {
       margin: 8px 0 4px 0; border: 1px solid #2a3a2a; border-radius: 8px;
       background: #0f1a0f; overflow: hidden; font-size: 13px;
}
.think-header {
       display: flex; align-items: center; gap: 8px;
       padding: 8px 14px; cursor: pointer; color: #6dbf6d;
       user-select: none;
}
.think-header:hover { background: #162516; }
.think-icon { font-size: 16px; }
.think-toggle { margin-left: auto; font-size: 11px; transition: transform .2s; }
.think-toggle.think-done { color: #444; }
.think-body {
       max-height: 0; overflow: hidden; transition: max-height .3s ease;
       padding: 0 14px; color: #5a8a5a; white-space: pre-wrap;
       font-family: monospace; font-size: 12px; line-height: 1.5;
}
.think-block.open .think-body { max-height: 600px; padding: 8px 14px 12px; }
.think-block.open .think-toggle:not(.think-done) { transform: rotate(180deg); }
.think-dots {
       display: inline-flex; gap: 5px; padding: 11px 14px;
       background: #161616; border: 1px solid #222; border-radius: 16px; border-bottom-left-radius: 4px;
}
.think-dots span {
       width: 6px; height: 6px; background: #444; border-radius: 50%;
       animation: dot-bounce 1.2s ease-in-out infinite;
}
.think-dots span:nth-child(2) { animation-delay: .18s; }
.think-dots span:nth-child(3) { animation-delay: .36s; }
@keyframes dot-bounce { 0%,80%,100%{transform:scale(.6);opacity:.4} 40%{transform:scale(1);opacity:1} }
/* actions */
.acts { display: flex; gap: 4px; margin-top: 7px; opacity: 0; transition: opacity .15s; }
.row:hover .acts { opacity: 1; }
.act {
       background: none; border: 1px solid #1e1e1e; color: #555; border-radius: 6px;
       padding: 4px 9px; font-size: 11px; cursor: pointer;
       display: flex; align-items: center; gap: 4px; transition: all .12s;
}
.act:hover { background: #1e1e1e; color: #e8e8e8; border-color: #333; }
.act.ok { color: #22c55e; border-color: #22c55e; }
.act.continue { color: #f59e0b; border-color: #f59e0b; }
.act.continue:hover { background: #2a1f0f; color: #fbbf24; border-color: #fbbf24; }
/* THINKING INDICATOR - INLINE IN STATS BAR */
#thinking-indicator {
       display: none;
       background: none;
       border: none;
       padding: 0;
       margin: 0;
       color: #a78bfa;
       font-size: 11px;
       align-items: center;
       gap: 4px;
       flex-shrink: 0;
       white-space: nowrap;
       order: -1;
}
#thinking-indicator.show {
       display: inline-flex;
}
#thinking-indicator .dots {
       display: inline-flex;
       gap: 2px;
       align-items: center;
}
#thinking-indicator .dots span {
       width: 3px;
       height: 3px;
       background: #a78bfa;
       border-radius: 50%;
       animation: dot-think 1s ease-in-out infinite;
}
#thinking-indicator .dots span:nth-child(2) {
       animation-delay: 0.15s;
}
#thinking-indicator .dots span:nth-child(3) {
       animation-delay: 0.3s;
}
@keyframes dot-think {
       0%, 60%, 100% {
         transform: scale(0.6);
         opacity: 0.5;
       }
       30% {
         transform: scale(1);
         opacity: 1;
       }
}
/* WORK INDICATOR: shown from 'send' until the first visible token */
#work-row { display: flex; padding: 6px 18px 2px; }
#work-row .work-card {
  display: inline-flex; align-items: center; gap: 12px; min-width: 190px;
  background: #161616; border: 1px solid #222; border-radius: 16px;
  border-bottom-left-radius: 4px; padding: 10px 14px; color: #bdbdbd; font-size: 12px;
}
#work-row .think-dots { padding: 0; background: none; border: none; }
#work-row .think-dots span { background: #a78bfa; width: 7px; height: 7px; }
#work-row .work-meta { display: flex; flex-direction: column; gap: 5px; min-width: 150px; }
#work-row .work-label { font-weight: 600; letter-spacing: .01em; }
#work-row .work-sub { color: #777; font-variant-numeric: tabular-nums; }
#work-row .work-bar { height: 3px; border-radius: 3px; background: #262626; overflow: hidden; }
#work-row .work-bar > i { display: block; height: 100%; width: 0%; background: #a78bfa; transition: width .6s ease; }
#work-row .work-bar.indet > i { width: 35%; animation: work-slide 1.4s ease-in-out infinite; }
@keyframes work-slide { 0% { margin-left: -35%; } 100% { margin-left: 100%; } }
@media (prefers-reduced-motion: reduce) {
  #work-row .think-dots span, #work-row .work-bar.indet > i { animation: none; }
}
/* THINKING MODE TOGGLE IN TOPBAR */
#btn-thinking {
       background: none; border: 1px solid #222; color: #555; border-radius: 7px;
       padding: 5px 9px; font-size: 12px; cursor: pointer; transition: all .12s;
       white-space: nowrap;
}
#btn-thinking:hover, #btn-thinking.on { background: #1e1535; color: #a78bfa; border-color: #4c1d95; }
/* RETRY BUTTON STYLING */
.act.retry { color: #3b82f6; }
.act.retry:hover { border-color: #3b82f6; background: #1e2847; }
/* ═══════════════════════════════════════ STATS BAR */
#stats {
       height: 28px; background: #0d0d0d; border-top: 1px solid #161616;
       display: flex; align-items: center; padding: 0 18px; gap: 18px;
       font-size: 11px; color: #333; flex-shrink: 0; font-variant-numeric: tabular-nums;
       flex-wrap: nowrap;
}
#stats .sv { color: #555; transition: color .3s; }
#stats.live .sv { color: #888; }
#stats-right { margin-left: auto; font-size: 10px; color: #333; display: flex; align-items: center; gap: 10px; }
/* token usage bar */
.token-bar {
       display: inline-flex; align-items: center; gap: 8px;
       background: #161616; border-radius: 12px; padding: 2px 8px;
       font-size: 10px;
}
.token-bar .fill {
       width: 50px; height: 4px; background: #2a2a2a; border-radius: 2px; overflow: hidden;
}
.token-bar .fill > div {
       height: 100%; background: #7c3aed; width: 0%; transition: width .2s;
}
/* Instant TPS indicator */
#instant-tps {
       display: none;
       background: #1e1e1e;
       border-radius: 12px;
       padding: 2px 8px;
       font-size: 10px;
       color: #a78bfa;
       white-space: nowrap;
}
#instant-tps.show {
       display: inline-flex;
       align-items: center;
       gap: 4px;
}
/* ═══════════════════════════════════════ COMPOSER */
#composer { border-top: 1px solid #161616; padding: 12px 18px 14px; background: #0d0d0d; flex-shrink: 0; }
#sys-wrap {
       max-width: var(--content-max-w); margin: 0 auto 10px; display: none;
       animation: fadein .15s ease;
}
#sys-wrap.show { display: block; }
#sys-in {
       width: 100%; background: #161616; border: 1px solid #222; color: #e8e8e8;
       border-radius: 9px; padding: 9px 12px; font-size: 13px; font-family: inherit;
       resize: vertical; min-height: 56px; outline: none; line-height: 1.5;
}
#sys-in:focus { border-color: #7c3aed; }
#sys-in::placeholder { color: #333; }
#input-row {
       max-width: var(--content-max-w); margin: 0 auto;
       display: flex; gap: 8px; align-items: flex-end;
       background: #161616; border: 1px solid #222; border-radius: 14px;
       padding: 9px 12px 9px 16px; transition: border-color .15s;
}
#input-row:focus-within { border-color: #7c3aed; box-shadow: 0 0 0 2px rgba(124,58,237,.12); }
#prompt {
       flex: 1; background: none; border: none; outline: none;
       color: #e8e8e8; font-size: 15px; font-family: inherit;
       resize: none; min-height: 22px; max-height: 200px; overflow-y: auto;
       line-height: 1.55;
}
#prompt::placeholder { color: #333; }
.btn-icon {
       flex-shrink: 0; border: none; border-radius: 9px;
       width: 37px; height: 37px; display: flex; align-items: center; justify-content: center;
       cursor: pointer; transition: all .15s; font-size: 16px;
}
#btn-send-stop {
       background: #7c3aed;
       color: #fff;
}
#btn-send-stop:hover:not(:disabled) {
       background: #6d28d9;
       transform: scale(1.06);
}
#btn-send-stop:disabled {
       background: #1e1e1e;
       color: #333;
       cursor: not-allowed;
       transform: none;
}
#send-icon, #stop-icon {
       width: 16px;
       height: 16px;
       fill: currentColor;
}
.hint { max-width: var(--content-max-w); margin: 6px auto 0; font-size: 11px; color: #282828; text-align: center; }
/* ═══════════════════════════════════════ TOAST NOTIFICATIONS */
#toast-container {
       position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
       z-index: 10000; display: flex; flex-direction: column; gap: 8px; pointer-events: none;
}
.toast {
       padding: 10px 18px; border-radius: 8px; font-size: 13px; font-family: inherit;
       pointer-events: auto; animation: slideUp .25s ease; max-width: 480px; text-align: center;
       box-shadow: 0 4px 20px rgba(0,0,0,0.5);
}
.toast.info    { background: #1e1535; color: #a78bfa; border: 1px solid #7c3aed; }
.toast.success { background: #1a2c1a; color: #22c55e; border: 1px solid #22c55e; }
.toast.warning { background: #2c1f0f; color: #f59e0b; border: 1px solid #f59e0b; }
.toast.error   { background: #2c1a1a; color: #f87171; border: 1px solid #ef4444; }
@keyframes slideUp { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:none} }
/* ═══════════════════════════════════════ SETTINGS MODAL */
#settings-modal {
       display: none;
       position: fixed; top: 0; left: 0; width: 100%; height: 100%;
       background: rgba(0,0,0,0.85);
       backdrop-filter: blur(4px);
       z-index: 1000;
       align-items: center;
       justify-content: center;
}
#settings-modal.show { display: flex; }
#settings-modal .modal-box {
       background: #111;
       border: 1px solid #333;
       border-radius: 16px;
       width: 450px;
       max-width: 90%;
       padding: 24px;
       box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
}
#settings-modal h3 {
       margin: 0 0 20px 0;
       color: #a78bfa;
       font-weight: 600;
       font-size: 1.4rem;
}
#settings-modal label {
       display: block;
       margin-bottom: 8px;
       font-size: 13px;
       color: #aaa;
}
#settings-modal input[type="password"] {
       width: 100%;
       background: #1e1e1e;
       border: 1px solid #333;
       color: #e8e8e8;
       border-radius: 8px;
       padding: 10px 12px;
       font-size: 13px;
       font-family: monospace;
       outline: none;
}
#settings-modal input[type="password"]:focus {
       border-color: #7c3aed;
}
#settings-modal .help-text {
       font-size: 11px;
       color: #666;
       margin-top: 6px;
}
#settings-modal .key-stats {
       background: #1a1a1a;
       border-radius: 8px;
       padding: 10px 12px;
       font-size: 12px;
       font-family: monospace;
       color: #aaa;
       margin-top: 12px;
}
#settings-modal .modal-actions {
       display: flex;
       gap: 12px;
       justify-content: flex-end;
       margin-top: 20px;
}
#settings-modal .btn-primary {
       background: #7c3aed;
       border: none;
       color: white;
       padding: 8px 16px;
       border-radius: 8px;
       cursor: pointer;
       font-weight: 500;
}
#settings-modal .btn-primary:hover { background: #6d28d9; }
#settings-modal .btn-secondary {
       background: #222;
       border: 1px solid #333;
       color: #ccc;
       padding: 8px 16px;
       border-radius: 8px;
       cursor: pointer;
}
#settings-modal .btn-secondary:hover { background: #2a2a2a; }
/* ═══════════════════════════════════════ RESPONSIVE */
@media (max-width: 600px) {
       #sidebar { display: none; }
       .cards { grid-template-columns: 1fr; }
       .bub { max-width: 96%; }
       #topbar { padding: 0 10px; gap: 6px; }
       #model-sel { max-width: 130px; }
       #settings-modal .modal-box { width: 95%; }
}
</style>
</head>
<body>
<!-- ════════════════════════════════════ SIDEBAR -->
<div id="sidebar">
       <div class="sb-head">
         <span class="sb-logo">⚡ Local LLM Cloud</span>
         <button id="btn-new" onclick="newChat()">
           <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor"><path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/></svg>
           New
         </button>
       </div>
       <div class="sb-search">
         <input id="hist-search" type="text" placeholder="Search conversations…" oninput="filterHistory()">
       </div>
       <div id="hist-list"></div>
       <div class="sb-foot">
         <div class="sb-stat" id="sb-stat">Connecting…</div>
         <!-- Live hardware-tier readout populated by applyRuntimeLimits() -->
         <div class="sb-stat" id="sb-limits" style="font-size:10px;color:#444;margin-top:2px"></div>
         <!-- export/import buttons -->
         <div style="margin-top:4px;display:flex;gap:4px">
           <button onclick="exportConv()" style="background:#1e1e1e;border:1px solid #333;color:#888;border-radius:5px;padding:2px 8px;font-size:10px;cursor:pointer" title="Export current conversation as JSON">Export</button>
           <button onclick="importConv()" style="background:#1e1e1e;border:1px solid #333;color:#888;border-radius:5px;padding:2px 8px;font-size:10px;cursor:pointer" title="Import a conversation from JSON file">Import</button>
         </div>
       </div>
</div>
<!-- ════════════════════════════════════ MAIN -->
<div id="main">
       <!-- TOPBAR -->
       <div id="topbar">
         <button id="btn-sb" onclick="toggleSb()" title="Toggle sidebar">☰</button>
         <span id="tb-title">New Conversation</span>
         <select id="model-sel" onchange="saveModel()" title="Select model">
           <option value="">Loading models...</option>
         </select>
         <div class="status-badge" id="status-badge" onclick="showStatusHelp()" title="Click for details">
           <div class="dot" id="dot"></div>
           <span id="status-txt">Connecting…</span>
         </div>
         <button id="btn-sys" onclick="toggleSys()" title="System prompt">⚙ System</button>
         <button id="btn-thinking" onclick="toggleThinkingMode()" title="Toggle thinking mode">💭 Thinking OFF</button>
         <select id="timeout-sel" class="timeout-select" title="Idle timeout: abort if no data arrives for this long (client-side only)">
           <option value="60">1 min</option>
           <option value="120">2 min</option>
           <option value="300" selected>5 min</option>
           <option value="600">10 min</option>
           <option value="0">No timeout</option>
         </select>
         <button id="btn-theme" onclick="toggleTheme()" title="Toggle dark/light theme">🌓</button>
         <button id="btn-settings" onclick="showSettings()" title="API Key Settings">⚙ Key</button>
         <!-- Internal models toggle -->
         <button id="btn-internal" onclick="toggleInternalModels()" title="Show internal/draft models for debugging">🔍 Internal OFF</button>
       </div>
       <!-- OFFLINE BANNER -->
       <div id="offline-bar"></div>
       <!-- CHAT -->
       <div id="chat"></div>
       <!-- Thinking Indicator (inline) -->
       <div id="thinking-indicator">
         💭 Model thinking<span class="dots"><span></span><span></span><span></span></span>
       </div>
       <!-- STATS BAR -->
       <div id="stats">
         <span>⚡ <span class="sv" id="st-tps">—</span> tok/s</span>
         <span>⏱ <span class="sv" id="st-ttft">—</span> TTFT</span>
         <span>📨 <span class="sv" id="st-tok">0</span> tokens</span>
         <span>⏳ <span class="sv" id="st-ela">—</span></span>
         <div id="stats-right"></div>
       </div>
       <!-- COMPOSER -->
       <div id="composer">
         <div id="sys-wrap">
           <textarea id="sys-in" placeholder="System prompt — sets the AI personality and behaviour (optional). Persists across messages."></textarea>
         </div>
         <div id="input-row">
           <textarea id="prompt" rows="1" placeholder="Message the AI… (Enter to send, Shift+Enter for new line)" maxlength="32000"></textarea>
           <button class="btn-icon" id="btn-send-stop" onclick="sendOrStop()" title="Send message or stop generation">
             <svg id="send-icon" viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
             <svg id="stop-icon" style="display:none;" viewBox="0 0 24 24"><path d="M6 4h12v12H6z"/></svg>
           </button>
         </div>
         <div class="hint">Enter to send · Shift+Enter for new line · Esc to stop · Ctrl+E export · Ctrl+I import</div>
       </div>
</div>
<!-- ════════════════════════════════════ SETTINGS MODAL -->
<div id="settings-modal">
       <div class="modal-box">
         <h3>🔑 API Key Settings</h3>
         <div style="margin-bottom:20px;">
           <label for="api-key-input">API Key</label>
           <input id="api-key-input" type="password" placeholder="Enter your API key (min 10 chars)">
           <div class="help-text">Leave empty to keep current key. Key is stored locally in your browser.</div>
         </div>
         <div style="margin-bottom:20px;">
           <label for="temp-input">Temperature — <span id="temp-val">0.7</span></label>
           <input id="temp-input" type="range" min="0" max="2" step="0.1" value="0.7"
                  style="width:100%" oninput="onTempInput(this.value)">
           <div class="help-text">Lower = more focused/deterministic. Higher = more creative/varied. Applies to your next message.</div>
         </div>
         <div style="margin-bottom:20px;">
           <label>Rate Limits &amp; Usage</label>
           <div class="key-stats" id="key-stats">Loading...</div>
         </div>
         <div class="modal-actions">
          <button class="btn-secondary" id="close-settings-btn">Cancel</button>
          <button class="btn-primary" id="save-key-btn">Save</button>
         </div>
       </div>
</div>
<!-- Toast container -->
<div id="toast-container"></div>
<script>
// ════════════════════════════════════════════════════════════ DEBUG LOGGING
// All diagnostic output routes through log.warning so production logs surface
// it. There is no log.debug — see the standing rule for this release.
const log = {
       warn: (msg, err) => console.warn(`[UI] ${msg}`, err || ''),
       info: (msg, data) => console.info(`[UI] ${msg}`, data || ''),
       error: (msg, err) => console.error(`[UI] ${msg}`, err || '')
};
// ════════════════════════════════════════════════════════════ STATE
const BASE     = location.origin;
const LS_H     = 'llmc_hist_v6';
const LS_M     = 'llmc_model';
const LS_S     = 'llmc_sys';
const LS_KEY   = 'llmc_api_key';
const LS_THINK = 'llmc_thinking_mode';
const LS_TEMP  = 'llmc_temperature';
const LS_THEME = 'llmc_theme';
const LS_INTERNAL = 'llmc_show_internal';
const MAX_CONV = 50;
let busy    = false;
let ctrl    = null;
let currentStreamAborted = false;
let cid     = null;
let convs   = {};
let online  = false;
let modelContextLength = 2048;
let currentModelId = '';
let thinkingModeEnabled = false;
let currentTemperature = 0.7;
try {
       const storedTemp = parseFloat(localStorage.getItem(LS_TEMP));
       if (!isNaN(storedTemp) && storedTemp >= 0 && storedTemp <= 2) currentTemperature = storedTemp;
} catch(_) {}
let turnCounter = 0;
let turnRequests = {};
let serverAvgTps = 0;
let modelStates = {};
let modelRefreshInterval = null;
let apiKey = null;
try {
       const storedKey = localStorage.getItem(LS_KEY);
       if (storedKey) {
         apiKey = storedKey;
         log.info('API key loaded from localStorage');
       }
} catch (_) {}
if (!apiKey && window.__API_KEY__) {
       apiKey = window.__API_KEY__;
       log.info('API key loaded from server (static)');
}
if (!apiKey) {
       log.info('No API key configured - will use localhost bypass');
}
function getApiKey() { return apiKey || null; }
function setApiKey(key) {
       apiKey = key;
       if (key) {
         try {
           localStorage.setItem(LS_KEY, key);
           log.info('API key saved to localStorage');
         } catch (_) {
           log.warn('Could not save key to localStorage');
         }
       } else {
         try {
           localStorage.removeItem(LS_KEY);
           log.info('API key removed from localStorage');
         } catch (_) {}
       }
}
let interruptionInfo = null;
let showInternalModels = false;
try {
       const stored = localStorage.getItem(LS_INTERNAL);
       if (stored !== null) showInternalModels = JSON.parse(stored);
} catch (_) {}
let allModels = [];

// ════════════════════════════════════════════════════════════ RUNTIME LIMITS
// The server computes hardware+model-aware limits at
// /v1/runtime/limits and we cache them here so send() and resumeTurn()
// can size their chunks correctly instead of using hard-coded numbers.
//
// Defaults below mirror the server's conservative fallbacks so the UI
// works correctly even before the first fetch completes. Every hard-coded
// limit in the rest of this file is derived from runtimeLimits.
//
// Refresh cadence:
//   * once at init()
//   * every 60 s (background)
//   * whenever the model list changes (fetchModelsWithRetry success)
//   * whenever the user changes the selected model (saveModel)
//
// The user-facing idle-timeout dropdown is auto-set ONCE, to the closest
// option matching the server's recommendation. After that we never touch
// it again — a user's manual choice always wins.
const DEFAULT_LIMITS = {
  version: '1.0.0',
  tier: 'small',
  hardware_profile: 'unknown',
  backend: 'cpu',
  model_id: '',
  model_size_mb: 0,
  context_length: 2048,
  context_safety_tokens: 256,
  min_response_tokens: 64,
  segment_size: 512,
  max_segments: 200,
  recommended_response_tokens: 1024,
  recommended_resume_chunk: 192,
  recommended_idle_timeout_s: 300,
  recommended_n_parallel: 1,
  measured_tps: 0.0,
  target_tps: 8.0,
  effective_tps: 8.0,
  vram_total_mb: 0,
  vram_free_mb: 0,
  ram_total_mb: 0,
  ram_free_mb: 0,
};
let runtimeLimits = Object.assign({}, DEFAULT_LIMITS);
let _timeoutAutoApplied = false;
let _limitsFetchInFlight = false;

async function fetchLimits() {
  if (_limitsFetchInFlight) return;
  if (!online) {
    log.info('fetchLimits: skipped (offline)');
    return;
  }
  _limitsFetchInFlight = true;
  try {
    const headers = { 'Content-Type': 'application/json' };
    if (apiKey) headers['X-API-Key'] = apiKey;
    const resp = await fetch(BASE + '/v1/runtime/limits', {
      method: 'GET',
      headers,
      signal: AbortSignal.timeout(5000),
      cache: 'no-store',
    });
    if (!resp.ok) {
      log.warn('Runtime limits fetch HTTP', resp.status);
      return;
    }
    const data = await resp.json();
    if (!data || typeof data !== 'object') return;
    runtimeLimits = Object.assign({}, DEFAULT_LIMITS, data);
    log.info('Runtime limits:', runtimeLimits);
    applyRuntimeLimits();
  } catch (err) {
    log.warn('Runtime limits fetch failed:', err.message);
  } finally {
    _limitsFetchInFlight = false;
  }
}

// Apply the server-recommended idle timeout to the dropdown ONCE. Never
// overwrite a manual choice — that would be hostile. Called on every
// limits fetch, but guarded by _timeoutAutoApplied.
function applyRuntimeLimits() {
  if (!_timeoutAutoApplied) {
    const sel = document.getElementById('timeout-sel');
    if (sel) {
      const rec = Number(runtimeLimits.recommended_idle_timeout_s) || 300;
      const opts = Array.from(sel.options)
        .map(o => Number(o.value))
        .filter(v => v > 0);
      if (opts.length) {
        const closest = opts.reduce(
          (best, v) => Math.abs(v - rec) < Math.abs(best - rec) ? v : best,
          opts[0],
        );
        sel.value = String(closest);
        _timeoutAutoApplied = true;
        log.info(
          'Idle timeout auto-set to', closest,
          's (server recommended', rec, 's for tier', runtimeLimits.tier + ')',
        );
      }
    }
  }
  // Live readout in the sidebar
  const lm = document.getElementById('sb-limits');
  if (lm) {
    try {
      lm.innerHTML =
        '<span style="color:#666">' + esc(runtimeLimits.tier || '?') + '</span> · ' +
        (Number(runtimeLimits.effective_tps) || 0).toFixed(1) + ' tps · ' +
        'chunk=' + (Number(runtimeLimits.recommended_resume_chunk) || 0) + ' · ' +
        'ctx=' + (Number(runtimeLimits.context_length) || 0);
    } catch (_) {
      lm.textContent = '';
    }
  }
}

// ══════════════════════════════ HEARTBEAT FRAMES + WORK INDICATOR
// Server control frames: \x1f STATUS:{json} \x1e  (never part of model text).
class FrameFilter {
  constructor() { this.carry = ''; }
  push(s) {
    const buf = this.carry + s;
    this.carry = '';
    const frames = [];
    let out = '', i = 0;
    while (i < buf.length) {
      const o = buf.indexOf('\x1f', i);
      if (o < 0) { out += buf.slice(i); break; }
      out += buf.slice(i, o);
      const c = buf.indexOf('\x1e', o + 1);
      if (c < 0) {
        if (buf.length - o > 4096) { out += buf.slice(o); }
        else { this.carry = buf.slice(o); }
        break;
      }
      const body = buf.slice(o + 1, c);
      if (body.startsWith('STATUS:')) {
        try { frames.push(JSON.parse(body.slice(7))); } catch (e) { /* ignore */ }
      }
      i = c + 1;
    }
    return { text: out, frames };
  }
}
let _workTimer = null, _workT0 = 0, _workFrame = null;
function _fmtDur(s) { s = Math.max(0, Math.round(s)); return s < 60 ? s + 's' : Math.floor(s / 60) + 'm ' + (s % 60) + 's'; }
function showWorkIndicator(label) {
  hideWorkIndicator();
  const chat = document.getElementById('chat');
  if (!chat) return;
  const row = document.createElement('div');
  row.id = 'work-row';
  row.setAttribute('role', 'status');
  row.setAttribute('aria-live', 'polite');
  row.innerHTML =
    '<div class="work-card"><div class="think-dots"><span></span><span></span><span></span></div>' +
    '<div class="work-meta"><div class="work-label"></div><div class="work-bar indet"><i></i></div>' +
    '<div class="work-sub"></div></div></div>';
  chat.appendChild(row);
  chat.scrollTop = chat.scrollHeight;
  _workT0 = performance.now(); _workFrame = null;
  row.querySelector('.work-label').textContent = label || 'Working…';
  _workTimer = setInterval(updateWorkIndicator, 1000);
  updateWorkIndicator();
}
function updateWorkIndicator(frame) {
  if (frame) _workFrame = frame;
  const row = document.getElementById('work-row');
  if (!row) return;
  const f = _workFrame || {};
  const el = (performance.now() - _workT0) / 1000;
  const bar = row.querySelector('.work-bar'), fill = bar.querySelector('i');
  let label = 'Working…', sub = _fmtDur(el);
  if (f.phase === 'prefill' && f.total > 0) {
    const pct = Math.min(99, Math.round(100 * f.done / f.total));
    label = 'Reading your prompt… ' + pct + '%';
    sub = f.done + ' / ' + f.total + ' tokens' +
          (f.reused ? ' (' + f.reused + ' cached)' : '') + ' · ' + _fmtDur(el);
    bar.classList.remove('indet'); fill.style.width = pct + '%';
  } else if (f.phase === 'decode') {
    label = 'Starting to write…';
  } else if (f.phase === 'queued') {
    label = 'Waiting for the model…';
  }
  row.querySelector('.work-label').textContent = label;
  row.querySelector('.work-sub').textContent = sub;
}
function hideWorkIndicator() {
  if (_workTimer) { clearInterval(_workTimer); _workTimer = null; }
  const row = document.getElementById('work-row');
  if (row) row.remove();
}

// Canonical markers shared with the server. The ⚡ prefix is a non-standard
// glyph that cannot reasonably appear in model output.
const STREAM_MARKER_RE = /\[⚡(TRUNCATED|INTERRUPTED|STALLED):[^\]]*?tokens=(\d+)[^\]]*\]/;
function parseStreamMarker(text) {
    if (!text) return null;
    const m = String(text).match(STREAM_MARKER_RE);
    if (!m) return null;
    return { kind: m[1], tokens: parseInt(m[2], 10) };
}
function stripStreamMarker(text) {
    return String(text || '').replace(STREAM_MARKER_RE, '').trim();
}

// ════════════════════════════════════════════════════════════ THEME
let currentTheme = 'dark';
function applyTheme(t) {
       document.documentElement.setAttribute('data-theme', t);
       currentTheme = t;
       localStorage.setItem(LS_THEME, t);
}
function toggleTheme() {
       applyTheme(currentTheme === 'dark' ? 'light' : 'dark');
}
(function initTheme() {
       const saved = localStorage.getItem(LS_THEME);
       if (saved) applyTheme(saved);
       else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) applyTheme('light');
})();
// ════════════════════════════════════════════════════════════ TOAST
function toast(msg, type='info', duration=3000) {
       const c = document.getElementById('toast-container');
       const t = document.createElement('div');
       t.className = `toast ${type}`;
       t.textContent = msg;
       c.appendChild(t);
       setTimeout(() => { t.style.opacity='0'; t.style.transition='opacity .3s'; setTimeout(()=>t.remove(),300) }, duration);
}
// ════════════════════════════════════════════════════════════ MODAL MANAGER
const ModalManager = {
       modalOpen: false,
       init() {
         const modal = document.getElementById('settings-modal');
         const closeBtn = document.getElementById('close-settings-btn');
         const saveBtn = document.getElementById('save-key-btn');
         if (!modal || !closeBtn || !saveBtn) {
           log.error('Modal elements not found!');
           return;
         }
         modal.addEventListener('click', (e) => {
           if (e.target === modal) {
             this.close();
           }
         });
         closeBtn.addEventListener('click', () => {
           this.close();
         });
         saveBtn.addEventListener('click', () => {
           this.saveKey();
         });
         document.addEventListener('keydown', (e) => {
           if (e.key === 'Escape' && this.modalOpen) {
             this.close();
           }
         });
         log.info('ModalManager initialized');
       },
       open() {
         const modal = document.getElementById('settings-modal');
         if (!modal) return;
         this.modalOpen = true;
         modal.classList.add('show');
         const input = document.getElementById('api-key-input');
         if (input) {
           if (apiKey) {
             const masked = apiKey.length > 8
               ? apiKey.substring(0, 4) + '•'.repeat(apiKey.length - 8) + apiKey.substring(apiKey.length - 4)
               : '••••••••';
             input.value = masked;
             input.setAttribute('data-has-key', 'true');
             input.placeholder = 'Leave empty to keep current key';
           } else {
             input.value = '';
             input.removeAttribute('data-has-key');
             input.placeholder = 'Enter your API key (min 10 chars)';
           }
           input.focus();
           input.select();
         }
         this.loadKeyStats();
         const tempInput = document.getElementById('temp-input');
         const tempLabel = document.getElementById('temp-val');
         if (tempInput) tempInput.value = currentTemperature;
         if (tempLabel) tempLabel.textContent = currentTemperature.toFixed(1);
         log.info('Settings modal opened');
       },
       close() {
         const modal = document.getElementById('settings-modal');
         if (!modal) return;
         this.modalOpen = false;
         modal.classList.remove('show');
         log.info('Settings modal closed');
       },
       async saveKey() {
         const input = document.getElementById('api-key-input');
         const saveBtn = document.getElementById('save-key-btn');
         if (!input || !saveBtn) return;
         const newKey = input.value.trim();
         const hasExistingKey = input.getAttribute('data-has-key') === 'true';
         if (hasExistingKey && newKey.includes('•')) {
           this.close();
           return;
         }
         if (!newKey && hasExistingKey) {
           setApiKey(null);
           toast('✅ API key removed', 'success');
           this.close();
           return;
         }
         if (!newKey) {
           toast('❌ API Key cannot be empty', 'error');
           input.focus();
           return;
         }
         if (newKey.length < 10) {
           toast('❌ API Key too short (min 10 characters)', 'error');
           input.focus();
           return;
         }
         if (newKey.length > 256) {
           toast('❌ API Key too long (max 256 characters)', 'error');
           return;
         }
         saveBtn.disabled = true;
         const originalText = saveBtn.textContent;
         saveBtn.textContent = '⏳ Validating...';
         try {
           const resp = await fetch(BASE + '/v1/me', {
             method: 'GET',
             headers: { 'X-API-Key': newKey, 'Content-Type': 'application/json' },
             signal: AbortSignal.timeout(5000)
           });
           if (!resp.ok) {
             if (resp.status === 401 || resp.status === 403) {
               toast('❌ Invalid API Key (authentication failed)', 'error');
               input.focus();
               return;
             }
             throw new Error(`Server error: HTTP ${resp.status}`);
           }
           const data = await resp.json();
           if (!data || typeof data !== 'object') {
             throw new Error('Invalid server response');
           }
           setApiKey(newKey);
           toast('✅ API key saved and validated!', 'success');
           try {
             await loadModels();
             log.info('Models reloaded with new key');
           } catch (e) {
             log.warn('Models reload failed, but key is saved:', e);
           }
           this.close();
         } catch (err) {
           log.error('saveKey error:', err);
           if (err.name === 'AbortError') {
             toast('⏱ Validation timeout – key saved anyway (server unreachable)', 'warning');
             setApiKey(newKey);
             this.close();
           } else if (err.message.includes('HTTP')) {
             toast(`❌ ${err.message}`, 'error');
           } else {
             toast(`❌ Failed: ${err.message || 'Unknown error'}`, 'error');
           }
         } finally {
           saveBtn.disabled = false;
           saveBtn.textContent = originalText;
         }
       },
       async loadKeyStats() {
         const statsDiv = document.getElementById('key-stats');
         if (!statsDiv) return;
         if (!apiKey) {
           statsDiv.textContent = 'No key configured';
           return;
         }
         try {
           const resp = await fetch(BASE + '/v1/me', {
             headers: { 'X-API-Key': apiKey, 'Content-Type': 'application/json' },
             signal: AbortSignal.timeout(3000)
           });
           if (resp.ok) {
             const data = await resp.json();
             statsDiv.innerHTML = `
               Rate Limit: ${data.rate_limit_rps || 'N/A'} req/s<br>
               Capacity: ${data.rate_limit_capacity || 'N/A'}<br>
               Total Requests: ${data.total_requests || 0}<br>
               Total Tokens: ${data.total_tokens || 0}
             `;
           } else {
             statsDiv.textContent = 'Failed to load stats';
           }
         } catch (err) {
           statsDiv.textContent = 'Error loading stats';
         }
       }
};
function toggleInternalModels() {
       showInternalModels = !showInternalModels;
       try { localStorage.setItem(LS_INTERNAL, JSON.stringify(showInternalModels)); } catch(_) {}
       const btn = document.getElementById('btn-internal');
       if (btn) {
         btn.textContent = showInternalModels ? '🔍 Internal ON' : '🔍 Internal OFF';
         btn.classList.toggle('on', showInternalModels);
       }
       loadModels();
       toast(showInternalModels ? '✅ Internal models visible' : '✅ Internal models hidden', 'info', 2000);
}
function isDraft(modelInfo) {
       return modelInfo && modelInfo.tags && modelInfo.tags.includes('draft');
}
function filterUserModels(models) {
       if (showInternalModels) return models;
       return models.filter(m => !isDraft(m));
}
async function fetchMetrics() {
       if (!online) return;
       try {
         const headers = { 'Content-Type': 'application/json' };
         if (apiKey) headers['X-API-Key'] = apiKey;
         const resp = await fetch(BASE + '/v1/metrics', {
           method: 'GET',
           headers,
           signal: AbortSignal.timeout(5000)
         });
         if (resp.ok) {
           const data = await resp.json();
           let newAvg = null;
           if (data.engine && typeof data.engine.avg_tps === 'number') {
             newAvg = data.engine.avg_tps;
           } else if (typeof data.avg_tps === 'number') {
             newAvg = data.avg_tps;
           }
           if (newAvg !== null && newAvg > 0) {
             serverAvgTps = newAvg;
             const tpsSpan = document.getElementById('st-tps');
             if (tpsSpan) tpsSpan.textContent = serverAvgTps.toFixed(1);
           }
         }
       } catch (err) {
         log.warn('Metrics fetch failed:', err.message);
       }
}
function showInstantTps(instantTps) {
       let el = document.getElementById('instant-tps');
       if (!el) {
         const statsRight = document.getElementById('stats-right');
         if (!statsRight) return;
         el = document.createElement('div');
         el.id = 'instant-tps';
         el.innerHTML = '📈 <span id="instant-tps-val">0.0</span> tok/s';
         statsRight.prepend(el);
       }
       const valSpan = document.getElementById('instant-tps-val');
       if (valSpan) valSpan.textContent = instantTps.toFixed(1);
       el.classList.add('show');
}
function hideInstantTps() {
       const el = document.getElementById('instant-tps');
       if (el) el.classList.remove('show');
}
function onTempInput(val) {
       const v = Math.max(0, Math.min(2, parseFloat(val) || 0.7));
       currentTemperature = v;
       const label = document.getElementById('temp-val');
       if (label) label.textContent = v.toFixed(1);
       try { localStorage.setItem(LS_TEMP, String(v)); } catch(_) {}
}
function toggleThinkingMode() {
       thinkingModeEnabled = !thinkingModeEnabled;
       try {
         localStorage.setItem(LS_THINK, JSON.stringify(thinkingModeEnabled));
         log.info('Thinking mode saved to localStorage:', thinkingModeEnabled);
       } catch (e) {
         log.warn('Could not save thinking mode:', e);
       }
       const btn = document.getElementById('btn-thinking');
       if (btn) {
         if (thinkingModeEnabled) {
           btn.textContent = '💭 Thinking ON';
           btn.classList.add('on');
           btn.title = 'Model will show thinking/reasoning (Qwen3 only)';
         } else {
           btn.textContent = '💭 Thinking OFF';
           btn.classList.remove('on');
           btn.title = 'Model will not show thinking/reasoning';
         }
       }
       toast(
         thinkingModeEnabled
           ? '✅ Thinking mode ON - Model will show internal reasoning'
           : '✅ Thinking mode OFF - No internal reasoning',
         'info',
         2000
       );
}
function showThinkingIndicator() {
       const el = document.getElementById('thinking-indicator');
       if (el && thinkingModeEnabled && busy) {
         el.classList.add('show');
       }
}
function hideThinkingIndicator() {
       const el = document.getElementById('thinking-indicator');
       if (el) {
         el.classList.remove('show');
       }
}
function updateThinkingStats(parsed) {
       if (!parsed || !parsed.has_thinking) return;
       const statsRight = document.getElementById('stats-right');
       if (!statsRight) return;
       let thinkStats = statsRight.querySelector('#think-stats');
       if (!thinkStats) {
         thinkStats = document.createElement('span');
         thinkStats.id = 'think-stats';
         thinkStats.title = 'Model thinking tokens (internal reasoning)';
         statsRight.appendChild(thinkStats);
       }
       thinkStats.innerHTML = `💭 <span class="sv">${parsed.thinking_tokens}</span> think`;
}
function updateStats(response) {
       if (!response || typeof response !== 'object') return;
       const ttft_ms = response.ttft_ms || 0;
       const total_time_s = response.total_time_s || 0;
       const tokens = response.usage?.completion_tokens || response.tokens_gen || 0;
       const ttftEl = document.getElementById('st-ttft');
       if (ttftEl) ttftEl.textContent = ttft_ms > 0 ? Math.round(ttft_ms) + 'ms' : '—';
       const tokEl = document.getElementById('st-tok');
       if (tokEl) tokEl.textContent = tokens || 0;
       const elaEl = document.getElementById('st-ela');
       if (elaEl) elaEl.textContent = total_time_s > 0 ? total_time_s.toFixed(1) + 's' : '—';
       if (response.model_thinking_active || response.has_thinking) {
         showThinkingIndicator();
         setTimeout(hideThinkingIndicator, 3000);
       }
       if (response.thinking_tokens && response.thinking_tokens > 0) {
         updateThinkingStats({
           has_thinking: true,
           thinking_tokens: response.thinking_tokens,
           thinking_type: response.thinking_type || 'internal'
         });
       }
       const statsEl = document.getElementById('stats');
       if (statsEl) statsEl.classList.toggle('live', (response.tps || 0) > 0);
}
function storeTurnRequest(turnId, params) {
       turnRequests[turnId] = params;
}
function getTurnRequest(turnId) {
       return turnRequests[turnId] || null;
}
// ════════════════════════════════════════════════════════════ STORAGE
function save()  { try { localStorage.setItem(LS_H, JSON.stringify(convs)); } catch(_){} }
function load()  { try { const r=localStorage.getItem(LS_H); if(r) convs=JSON.parse(r); } catch(_){} }
function curConv() { return cid && convs[cid] ? convs[cid] : null; }
function ensureConv() {
       if (!cid || !convs[cid]) {
         cid = 'c' + Date.now();
         convs[cid] = { title: 'New Conversation', msgs: [], ts: Date.now() };
         save();
       }
       return convs[cid];
}
// ════════════════════════════════════════════════════════════ HISTORY UI
function renderHistory(filter='') {
       const ul = document.getElementById('hist-list');
       const fl = filter.toLowerCase();
       const items = Object.entries(convs)
         .filter(([,c]) => !fl || c.title.toLowerCase().includes(fl))
         .sort(([,a],[,b]) => b.ts - a.ts)
         .slice(0, MAX_CONV);
       if (!items.length) {
         ul.innerHTML = '<div style="padding:14px 16px;font-size:12px;color:#333;text-align:center">' +
           (fl ? 'No matches' : 'No conversations yet') + '</div>';
         return;
       }
       ul.innerHTML = items.map(([id, c]) => {
         const active = id === cid ? ' active' : '';
         const t = relTime(c.ts);
         return '<div class="hi'+active+'" data-id="'+id+'" onclick="loadConv(\''+id+'\')" title="'+esc(c.title)+'">' +
'<span class="hi-title">'+esc(c.title.slice(0,52))+'</span>' +
'<span class="hi-time">'+t+'</span>' +
'<span class="hi-del" onclick="delConv(event,\''+id+'\')" title="Delete">✕</span>' +
'</div>';
       }).join('');
}
function relTime(ts) {
       const diff = Date.now() - ts;
       if (diff < 60000)     return 'now';
       if (diff < 3600000)   return Math.floor(diff/60000) + 'm';
       if (diff < 86400000)  return Math.floor(diff/3600000) + 'h';
       return Math.floor(diff/86400000) + 'd';
}
function filterHistory() {
       renderHistory(document.getElementById('hist-search').value);
}
function loadConv(id) {
       if (busy) return;
       cid = id;
       const c = convs[id];
       if (!c) return;
       const chat = document.getElementById('chat');
       chat.innerHTML = '';
       c.msgs.forEach(m => appendMsg(m.role, m.content, false));
       document.getElementById('tb-title').textContent = c.title;
       renderHistory(document.getElementById('hist-search').value);
       chat.scrollTop = chat.scrollHeight;
}
function delConv(e, id) {
       e.stopPropagation();
       delete convs[id];
       save();
       if (id === cid) newChat();
       else renderHistory(document.getElementById('hist-search').value);
}
function newChat() {
       if (busy) return;
       cid = 'c' + Date.now();
       convs[cid] = { title: 'New Conversation', msgs: [], ts: Date.now() };
       save();
       showEmpty();
       document.getElementById('tb-title').textContent = 'New Conversation';
       renderHistory(document.getElementById('hist-search').value);
}
// ═══════════════════════════════════════ EXPORT / IMPORT
function exportConv() {
       const c = curConv();
       if (!c) { toast('No conversation to export','warning'); return; }
       const blob = new Blob([JSON.stringify(c,null,2)],{type:'application/json'});
       const a = document.createElement('a');
       a.href = URL.createObjectURL(blob);
       a.download = `conversation-${cid}.json`;
       a.click();
       toast('✅ Conversation exported','success',1500);
}
function importConv() {
       const input = document.createElement('input');
       input.type = 'file';
       input.accept = '.json';
       input.onchange = async () => {
         const file = input.files[0];
         if (!file) return;
         try {
           const text = await file.text();
           const data = JSON.parse(text);
           if (!data.msgs || !Array.isArray(data.msgs)) {
             toast('❌ Invalid conversation file','error');
             return;
           }
           const newId = 'c' + Date.now();
           convs[newId] = { title: data.title || 'Imported', msgs: data.msgs, ts: Date.now() };
           save();
           cid = newId;
           loadConv(newId);
           toast('✅ Conversation imported','success');
         } catch(e) {
           toast('❌ Failed to import: ' + e.message,'error');
         }
       };
       input.click();
}
// ════════════════════════════════════════════════════════════ EMPTY STATE
function showEmpty() {
       document.getElementById('chat').innerHTML =
'<div id="empty">'+
'<div class="emp-icon">⚡</div>'+
'<div class="emp-title">Local LLM Cloud</div>'+
'<div class="emp-sub">Your private AI, running entirely on your hardware. No cloud, no fees, no data leaves your machine.</div>'+
'<div class="cards">'+
'<div class="card" onclick="starter(this)"><span class="card-title">📝 Write & Edit</span>Draft emails, essays, or creative content</div>'+
'<div class="card" onclick="starter(this)"><span class="card-title">💻 Code Assistant</span>Debug, explain, refactor, or write code</div>'+
'<div class="card" onclick="starter(this)"><span class="card-title">🔍 Research & Analysis</span>Summarise, explain, or break down any topic</div>'+
'<div class="card" onclick="starter(this)"><span class="card-title">🛡 Security Agent</span>Analyse logs, assess threats, review configs</div>'+
'</div></div>';
}
function starter(card) {
       if (busy) return;
       const map = {
'Write & Edit':        `Help me write a professional email to my team about an upcoming sprint deadline. I need to communicate that we need to extend by one week.`,
'Code Assistant':      `Review this Python function and suggest improvements:
def process(data):
         result = []
         for i in range(len(data)):
             if data[i] > 0:
                 result.append(data[i] * 2)
         return result`,
'Research & Analysis': `Give me a comprehensive explanation of how large language models work — covering tokenization, transformer architecture, attention mechanisms, and training. Make it detailed enough for a software engineer.`,
'Security Agent':      `Analyse these server logs for security threats and provide a detailed threat assessment:
2024-01-15 03:42:17 WARN Failed login from 192.168.1.105 user=admin
2024-01-15 03:42:19 WARN Failed login from 192.168.1.105 user=root
2024-01-15 03:42:21 WARN Failed login from 192.168.1.105 user=administrator`,
       };
       const title = card.querySelector('.card-title').textContent.replace(/[📝💻🔍🛡]/g,'').trim();
       document.getElementById('prompt').value = map[title] || title;
       autoResize();
       send();
}
// ════════════════════════════════════════════════════════════ SERVER STATUS
let healthCheckRetries = 0;
const MAX_HEALTH_CHECK_RETRIES = 3;
async function checkStatus() {
       try {
         const r = await fetch(BASE + '/health', {
           signal: AbortSignal.timeout(4000),
           headers: { 'Content-Type': 'application/json' },
           cache: 'no-cache'
         });
         if (!r.ok) {
           log.warn('Health check HTTP error:', r.status);
           setOnline(false);
           return;
         }
         const d = await r.json();
         const isHealthy = d.status === 'ok';
         if (isHealthy) {
           window.__SERVER_LAST_SEEN__ = new Date().toLocaleTimeString();
           healthCheckRetries = 0;
         }
         setOnline(isHealthy);
         window.__SERVER_HEALTH_STATUS__ = d.health_status || 'unknown';
       } catch (err) {
         log.warn('Health check network error:', err.message);
         if (healthCheckRetries < MAX_HEALTH_CHECK_RETRIES) {
           healthCheckRetries++;
           setTimeout(() => checkStatus(), 2000 * healthCheckRetries);
         } else {
           setOnline(false);
           healthCheckRetries = 0;
         }
       }
}
function setOnline(v) {
       online = v;
       const dot = document.getElementById('dot');
       const txt = document.getElementById('status-txt');
       const bar = document.getElementById('offline-bar');
       if (v) {
         dot.className = 'dot on';
         txt.textContent = 'Online';
         if (bar) {
           bar.classList.remove('show');
           bar.innerHTML = '';
         }
       } else {
         dot.className = 'dot off';
         txt.textContent = 'Offline';
         if (bar) {
           const dismissId = `banner-${Date.now()}`;
           bar.innerHTML = `
             <span>⚠️ Cannot connect to the Local LLM Cloud server.</span>
             <button id="${dismissId}-retry">⟳ Retry</button>
             <button id="${dismissId}-dismiss">✕ Dismiss</button>
           `;
           bar.classList.add('show');
           document.getElementById(`${dismissId}-retry`)?.addEventListener('click', () => {
             checkStatus();
             toast('🔄 Checking connection...', 'info', 2000);
           });
           document.getElementById(`${dismissId}-dismiss`)?.addEventListener('click', () => {
             bar.classList.remove('show');
           });
         }
       }
       if (!busy) updateUIStates();
}
function showStatusHelp() {
       if (online) {
         if (Object.values(modelStates).some(s => s === 'loading')) {
           toast('🔄 Models are loading...', 'info', 3000);
         } else {
           toast('✅ Server is online and ready', 'success', 3000);
         }
       } else {
         toast('❌ Server offline. Please check your connection.', 'error', 6000);
       }
}
function updateUIStates() {
       const hasModels = Object.values(modelStates).some(s => s === 'ready');
       const canSend = online && hasModels && !busy;
       const btn = document.getElementById('btn-send-stop');
       const sendIcon = document.getElementById('send-icon');
       const stopIcon = document.getElementById('stop-icon');
       if (busy) {
         btn.title = 'Stop generation';
         sendIcon.style.display = 'none';
         stopIcon.style.display = 'block';
         btn.style.background = '#ef4444';
       } else {
         btn.title = 'Send message';
         sendIcon.style.display = 'block';
         stopIcon.style.display = 'none';
         btn.style.background = '#7c3aed';
       }
       btn.disabled = !online && !busy;
       document.getElementById('btn-sys').disabled = !online;
       document.getElementById('btn-settings').disabled = !online;
       document.getElementById('model-sel').disabled = !online;
       document.getElementById('timeout-sel').disabled = !online;
}
// ════════════════════════════════════════════════════════════ MODEL LOADING
let modelLoadController = null;
let modelLoading        = false;
async function loadModels() {
       if (modelLoadController) {
         modelLoadController.abort();
         await new Promise(resolve => setTimeout(resolve, 50));
       }
       if (modelLoading) return;
       modelLoading = true;
       try {
         await fetchModelsWithRetry(0);
       } finally {
         modelLoading = false;
         modelLoadController = null;
       }
}
async function fetchModelsWithRetry(attemptNum = 0, last503Time = 0) {
       const controller = new AbortController();
       modelLoadController = controller;
       const timeoutMs = attemptNum === 0 ? 12000 : Math.min(20000, 12000 + attemptNum * 2000);
       const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
       const headers = { "Content-Type": "application/json" };
       if (apiKey) headers["X-API-Key"] = apiKey;
       try {
         log.info(`Fetching models (attempt ${attemptNum + 1})...`);
         document.getElementById('model-sel').innerHTML = '<option value="">⏳ Loading...</option>';
         document.getElementById('sb-stat').innerHTML = '<div style="color:#f59e0b">Attempt ' + (attemptNum + 1) + '...</div>';
         const r = await fetch(BASE + '/v1/models', {
           signal: controller.signal,
           headers: headers,
           cache: 'no-cache'
         });
         if (r.ok) {
           const d = await r.json();
           const models = (d.data || []).filter(m => m.id);
           allModels = models;
           modelStates = {};
           models.forEach(m => { modelStates[m.id] = m.status || 'ready'; });
           renderModelSelect(models);
           const userModels = filterUserModels(models);
           const statsHtml = userModels.map(m =>
             `<div>${esc(m.id)} <span class="sv">${
               m.gpu_layers || 0}GPU / ${m.vram_mb || 0}MB${
               m.context_length ? ' · ctx=' + m.context_length : ''} (${m.status || 'ready'})</span></div>`
           ).join('') + `<div style="color:#333;font-size:10px;margin-top:4px">${new Date().toLocaleTimeString()}</div>`;
           document.getElementById('sb-stat').innerHTML = statsHtml;
           updateUIStates();
           stopModelRefresh();
           if (Object.values(modelStates).some(s => s === 'loading')) {
             startModelRefresh();
           }
           // Model inventory changed — limits may have moved too (loaded
           // model changed, or a new one swapped in).
           fetchLimits();
           return true;
         }
         if (r.status === 503) {
           const now = Date.now();
           if (last503Time > 0 && (now - last503Time) > 300000) {
             document.getElementById('model-sel').innerHTML = '<option value="">❌ Load timeout</option>';
             document.getElementById('sb-stat').innerHTML = '<div style="color:#f87171">Models loading timeout - check server</div>';
             return false;
           }
           last503Time = now;
           const waitMs = 30000;
           document.getElementById('model-sel').innerHTML = '<option value="">⏳ Models loading...</option>';
           document.getElementById('sb-stat').innerHTML = '<div style="color:#f59e0b">Waiting ' + (waitMs/1000) + 's...</div>';
           await new Promise(r => setTimeout(r, waitMs));
           if (modelLoadController && !modelLoadController.signal.aborted) {
             return fetchModelsWithRetry(0, last503Time);
           }
           return false;
         }
         if (!r.ok) {
           throw new Error(`HTTP ${r.status} ${r.statusText}`);
         }
       } catch (e) {
         if (e.name === 'AbortError') {
           log.warn(`Timeout (${timeoutMs}ms) — retrying...`);
           if (attemptNum < 3) {
             await new Promise(r => setTimeout(r, 1000 + attemptNum * 1000));
             return fetchModelsWithRetry(attemptNum + 1, last503Time);
           }
           document.getElementById('model-sel').innerHTML = '<option value="">❌ Load timeout</option>';
           document.getElementById('sb-stat').innerHTML = '<div style="color:#f87171">Load timeout - check server/network</div>';
           return false;
         }
         log.error('Model load error:', e.message);
         if (attemptNum < 3) {
           const backoff = Math.min(1000 * Math.pow(2, attemptNum), 16000);
           document.getElementById('model-sel').innerHTML = '<option value="">⏳ Retrying...</option>';
           document.getElementById('sb-stat').innerHTML = `<div style="color:#f59e0b">Attempt ${attemptNum+1}/3...</div>`;
           await new Promise(r => setTimeout(r, backoff));
           return fetchModelsWithRetry(attemptNum + 1, last503Time);
         }
         document.getElementById('model-sel').innerHTML = '<option value="">❌ Load failed</option>';
         document.getElementById('sb-stat').innerHTML = '<div style="color:#f87171">Load failed - refresh to retry</div>';
         return false;
       } finally {
         clearTimeout(timeoutId);
       }
}
function renderModelSelect(models) {
       const sel = document.getElementById('model-sel');
       if (!sel) return;
       const userModels = filterUserModels(models);
       const readyModels = userModels.filter(m => m.status === 'ready');
       const loadingModels = userModels.filter(m => m.status === 'loading');
       const errorModels = userModels.filter(m => m.status === 'error');
       let savedModel = localStorage.getItem(LS_M) || '';
       if (!showInternalModels) {
         const savedInfo = allModels.find(m => m.id === savedModel);
         if (savedInfo && isDraft(savedInfo)) savedModel = '';
       }
       let defaultModel = readyModels.find(m => m.id === savedModel);
       if (!defaultModel && readyModels.length > 0) defaultModel = readyModels[0];
       if (defaultModel && typeof defaultModel.context_length === 'number') {
         modelContextLength = defaultModel.context_length;
       }
       currentModelId = defaultModel ? defaultModel.id : '';
       let html = '';
       const hasAnyReady = readyModels.length > 0;
       if (hasAnyReady) {
         html += '<option value="__smart__">🤖 Auto-Router</option>';
       }
       readyModels.forEach(m => {
         const isDraftModel = isDraft(m);
         const suffix = (showInternalModels && isDraftModel) ? ' (draft)' : '';
         html += `<option value="${esc(m.id)}"${m.id === savedModel ? ' selected' : ''}>${esc(m.id)}${suffix} (ready)</option>`;
       });
       loadingModels.forEach(m => {
         html += `<option value="${esc(m.id)}" disabled class="loading">${esc(m.id)} (loading...)</option>`;
       });
       errorModels.forEach(m => {
         html += `<option value="${esc(m.id)}" disabled class="error">${esc(m.id)} (error)</option>`;
       });
       if (!hasAnyReady && loadingModels.length === 0) {
         html = '<option value="">❌ No models available</option>';
       }
       sel.innerHTML = html;
}
function startModelRefresh() {
       stopModelRefresh();
       modelRefreshInterval = setInterval(async () => {
         try {
           const resp = await fetch(BASE + '/v1/models', {
             signal: AbortSignal.timeout(5000),
             headers: apiKey ? { 'X-API-Key': apiKey } : {},
             cache: 'no-cache'
           });
           if (resp.ok) {
             const d = await resp.json();
             const models = (d.data || []).filter(m => m.id);
             allModels = models;
             const anyLoading = models.some(m => m.status === 'loading');
             modelStates = {};
             models.forEach(m => { modelStates[m.id] = m.status || 'ready'; });
             renderModelSelect(models);
             updateUIStates();
             if (!anyLoading) stopModelRefresh();
           }
         } catch (e) { /* ignore polling errors */ }
       }, 5000);
}
function stopModelRefresh() {
       if (modelRefreshInterval) {
         clearInterval(modelRefreshInterval);
         modelRefreshInterval = null;
       }
}
function saveModel() {
       const sel = document.getElementById('model-sel');
       const selectedId = sel.value;
       if (selectedId === '__smart__') {
         localStorage.setItem(LS_M, selectedId);
         updateUIStates();
         // Limits depend on the loaded model — refresh shortly after the
         // server has had time to notice the change.
         setTimeout(fetchLimits, 500);
         return;
       }
       if (!showInternalModels) {
         const modelInfo = allModels.find(m => m.id === selectedId);
         if (modelInfo && isDraft(modelInfo)) {
           toast('⚠️ Draft models are hidden. Enable "Internal" to select this model.', 'warning', 4000);
           const prev = localStorage.getItem(LS_M);
           const prevInfo = allModels.find(m => m.id === prev);
           if (prev && prev !== selectedId && (!prevInfo || !isDraft(prevInfo))) {
             sel.value = prev;
           } else {
             const firstReady = allModels.find(m => m.status === 'ready' && !isDraft(m));
             sel.value = firstReady ? firstReady.id : '__smart__';
           }
           updateUIStates();
           return;
         }
       }
       localStorage.setItem(LS_M, selectedId);
       updateUIStates();
       // Limits depend on the loaded model — refresh shortly after the
       // server has had time to notice the change.
       setTimeout(fetchLimits, 500);
}
// ════════════════════════════════════ TEXT FILTERING
// FIX-2: this runs on ONE whitespace-delimited token at a time. It must only
// strip control tokens. The old "^(Therefore|Thus|So|Hence...)" rule (no word
// boundary, case-insensitive) deleted every word starting with "so"/"thus"
// (Some, software, solution, sort, sound, social ...) and the "^I should" /
// "^Let me" / "^Final Answer" rules were dead code or corrupting.
function filterThinkingAndMetaText(chunk) {
    chunk = chunk.replace(/\[end of text\]/gi, '');
    chunk = chunk.replace(/<\|(?:eot_id|end_of_text|begin_of_text|end_header_id|end|user|assistant|system|im_start|im_end|endoftext)\|>/g, '');
    chunk = chunk.replace(/<\|start_header_id\|>[^<]*<\|end_header_id\|>/g, '');
    if (chunk.trim() === '⏸ Stopped by user' || chunk.trim() === 'Stopped by user') {
        return '';
    }
    return chunk;
}
// ════════════════════════════════════════ ROBUST STREAM PARSER
// FIX-3: the old push() found "<think>" but never consumed it, so `continue`
// re-ran forever (tab freeze). State machine below always consumes the tag it
// matched, so every loop iteration makes progress.
//
// FIX-3b (this release): stray '</think>' with no matching opener — the exact
// leak the Qwen3-4B model produces after the server's empty think seed — is
// now consumed silently instead of being forwarded as ordinary content.
class RobustStreamParser {
    constructor() {
        this.buffer = '';
        this.inThinkingBlock = false;
    }
    static _find(buf, tags) {
        let best = -1, len = 0;
        for (const t of tags) {
            const i = buf.indexOf(t);
            if (i >= 0 && (best < 0 || i < best)) { best = i; len = t.length; }
        }
        return best < 0 ? null : { idx: best, len };
    }
    push(chunk) {
        this.buffer += chunk;
        const results = [];
        while (this.buffer.length > 0) {
            if (this.inThinkingBlock) {
                const c = RobustStreamParser._find(this.buffer, ['</think>', '</thinking>']);
                if (c) {
                    this.buffer = this.buffer.substring(c.idx + c.len);
                    this.inThinkingBlock = false;
                    continue;
                }
                // No close tag yet: discard thinking text, keep only a tail that
                // could be the start of a split close tag.
                if (this.buffer.length > 16) this.buffer = this.buffer.slice(-16);
                break;
            }
            // Stray closer with no matching opener (the exact leak the
            // 4B model produces after a malformed non-thinking seed).
            // Drop it silently instead of leaking '</think>' to the user.
            const orphan = RobustStreamParser._find(this.buffer, ['</think>', '</thinking>']);
            const opener = RobustStreamParser._find(this.buffer, ['<think>', '<thinking>']);
            if (orphan && (!opener || orphan.idx < opener.idx)) {
                if (orphan.idx > 0) results.push(this.buffer.substring(0, orphan.idx));
                this.buffer = this.buffer.substring(orphan.idx + orphan.len).replace(/^\s+/, '');
                continue;
            }
            const o = RobustStreamParser._find(this.buffer, ['<think>', '<thinking>']);
            if (o) {
                if (o.idx > 0) results.push(this.buffer.substring(0, o.idx));
                this.buffer = this.buffer.substring(o.idx + o.len);   // consume the tag
                this.inThinkingBlock = true;
                continue;
            }
            const spaceIdx = this.buffer.search(/\s/);
            if (spaceIdx === -1) {
                if (this.buffer.length > 100) {
                    results.push(this.buffer);
                    this.buffer = '';
                } else {
                    break;
                }
            } else {
                const token = this.buffer.substring(0, spaceIdx);
                const rest = this.buffer.substring(spaceIdx).match(/^(\s+)/)[0];
                if (token) results.push(token);
                results.push(rest);
                this.buffer = this.buffer.substring(spaceIdx + rest.length);
            }
        }
        return results.filter(t => t !== '');
    }
    flush() {
        // Never emit text that is still inside an unclosed thinking block.
        const out = (!this.inThinkingBlock && this.buffer) ? [this.buffer] : [];
        this.buffer = '';
        return out;
    }
}
// ════════════════════════════════════════ TOKEN ESTIMATION
// FIX (resume audit): /3.5 under-counted by ~15% versus the server's own
// fallback of /2.5 and the real Qwen tokenizer (~3.0 chars/token on mixed
// English/code). That mismatch caused the resume path to under-estimate
// prompt cost and over-request max_tokens, which the server then had to
// trim. /3.0 matches measured throughput on this class of prompts.
function estimateTokens(text) {
       return Math.max(1, Math.ceil((text || '').length / 3.0));
}
function updateTokenIndicator(used, total) {
       const rightSpan = document.getElementById('stats-right');
       if (!rightSpan) return;
       // Preserve any instant-tps child if it exists
       const instant = rightSpan.querySelector('#instant-tps');
       const pct = total ? (used / total) * 100 : 0;
       const warn = pct > 85 ? '#ef4444' : (pct > 70 ? '#f59e0b' : '#7c3aed');
       rightSpan.innerHTML = `<div class="token-bar" title="Token usage: ${used} / ${total}">
         <span>${used}/${total}</span>
         <div class="fill"><div style="width:${Math.min(100, pct)}; background:${warn}"></div></div>
       </div>`;
       if (instant) rightSpan.prepend(instant);
}
// ════════════════════════════════════════════════════════════ MARKDOWN & SANITIZATION
function esc(s) {
       return String(s||'')
         .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
         .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}
function sanitizeAssistantResponse(raw) {
       let cleaned = raw;
       const userAssistantPattern = /^User:[\s\S]*?Assistant:/i;
       if (userAssistantPattern.test(cleaned)) {
         const parts = cleaned.split(/Assistant:/i);
         if (parts.length > 1) {
           cleaned = parts.slice(1).join('Assistant:').trim();
         }
       }
       cleaned = cleaned.replace(/^Assistant:\s*/i, '');
       return cleaned;
}
// ─── Syntax highlighting with explicit languages and generic fallback ──
const HL_RULES = {
       python: {
         comment: /#.*/,
         string: /(\x22\x22\x22[\s\S]*?\x22\x22\x22|\x27\x27\x27[\s\S]*?\x27\x27\x27|f?"(?:[^"\\]|\\.)*"|f?'(?:[^'\\]|\\.)*')/,
         number: /\b\d+\.?\d*\b/,
         keyword: /\b(?:def|class|import|from|as|return|if|elif|else|for|while|try|except|finally|with|yield|lambda|pass|break|continue|and|or|not|in|is|None|True|False|self|async|await|raise|global|nonlocal|del|assert)\b/,
         fn: /\b([A-Za-z_]\w*)(?=\()/,
       },
       javascript: {
         comment: /\/\/.*|\/\*[\s\S]*?\*\//,
         string: /(`(?:[^`\\]|\\.)*`|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/,
         number: /\b\d+\.?\d*\b/,
         keyword: /\b(?:function|const|let|var|return|if|else|for|while|try|catch|finally|class|extends|new|this|import|export|from|as|default|async|await|yield|typeof|instanceof|in|of|null|undefined|true|false|switch|case|break|continue|throw|static|get|set)\b/,
         fn: /\b([A-Za-z_$]\w*)(?=\()/,
       },
       ruby: {
         comment: /#.*/,
         string: /("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|%[qwQWRr]?[^a-zA-Z0-9\s][\s\S]*?[^a-zA-Z0-9\s])/,
         number: /\b\d+\.?\d*\b/,
         keyword: /\b(?:def|class|module|require|include|extend|attr_accessor|attr_reader|attr_writer|if|elsif|else|unless|case|when|for|while|until|begin|rescue|ensure|raise|return|yield|super|self|nil|true|false|and|or|not|in|do|end)\b/,
         fn: /\b([A-Za-z_]\w*)(?=\()/,
       },
       go: {
         comment: /\/\/.*|\/\*[\s\S]*?\*\//,
         string: /(`(?:[^`\\]|\\.)*`|"(?:[^"\\]|\\.)*")/,
         number: /\b\d+\.?\d*\b/,
         keyword: /\b(?:package|import|func|var|const|type|struct|interface|map|chan|go|select|defer|if|else|for|range|switch|case|default|break|continue|fallthrough|return|goto|make|new|append|copy|len|cap|close|delete|panic|recover|interface|nil|true|false|iota)\b/,
         fn: /\b([A-Za-z_]\w*)(?=\()/,
       },
       rust: {
         comment: /\/\/.*|\/\*[\s\S]*?\*\//,
         string: /("[^"\\]*(?:\\.[^"\\]*)*"|'[^'\\]*(?:\\.[^'\\]*)*')/,
         number: /\b\d+\.?\d*\b/,
         keyword: /\b(?:fn|pub|struct|enum|impl|trait|use|mod|extern|static|mut|let|match|if|else|for|while|loop|return|break|continue|async|await|unsafe|macro_rules|type|where|self|Self|super|true|false|None|Some|Ok|Err)\b/,
         fn: /\b([A-Za-z_]\w*)(?=\()/,
       },
       c: {
         comment: /\/\/.*|\/\*[\s\S]*?\*\//,
         string: /"(?:[^"\\]|\\.)*"/,
         number: /\b\d+\.?\d*\b/,
         keyword: /\b(?:int|char|float|double|void|long|short|unsigned|signed|const|volatile|static|extern|inline|restrict|struct|union|enum|typedef|sizeof|if|else|for|while|do|switch|case|default|break|continue|return|goto|main)\b/,
         fn: /\b([A-Za-z_]\w*)(?=\()/,
       },
       cpp: {
         comment: /\/\/.*|\/\*[\s\S]*?\*\//,
         string: /"(?:[^"\\]|\\.)*"/,
         number: /\b\d+\.?\d*\b/,
         keyword: /\b(?:int|char|float|double|void|long|short|unsigned|signed|const|volatile|static|extern|inline|restrict|struct|union|enum|typedef|class|namespace|using|template|typename|this|new|delete|friend|virtual|override|final|public|private|protected|if|else|for|while|do|switch|case|default|break|continue|return|goto|main)\b/,
         fn: /\b([A-Za-z_]\w*)(?=\()/,
       },
       java: {
         comment: /\/\/.*|\/\*[\s\S]*?\*\//,
         string: /"(?:[^"\\]|\\.)*"/,
         number: /\b\d+\.?\d*\b/,
         keyword: /\b(?:class|interface|enum|extends|implements|import|package|public|private|protected|static|final|abstract|synchronized|volatile|transient|native|strictfp|if|else|for|while|do|switch|case|default|break|continue|return|throw|throws|try|catch|finally|new|this|super|instanceof|boolean|byte|char|short|int|long|float|double|void|true|false|null)\b/,
         fn: /\b([A-Za-z_]\w*)(?=\()/,
       },
       php: {
         comment: /\/\/.*|\/\*[\s\S]*?\*\//,
         string: /("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/,
         number: /\b\d+\.?\d*\b/,
         keyword: /\b(?:function|class|interface|trait|use|namespace|extends|implements|public|private|protected|static|final|abstract|if|else|elseif|for|foreach|while|do|switch|case|default|break|continue|return|throw|try|catch|finally|new|this|parent|self|echo|print|die|exit|isset|unset|empty|array|list|callable|int|float|string|bool|true|false|null)\b/,
         fn: /\b([A-Za-z_]\w*)(?=\()/,
       },
       json: {
         comment: /(?!)x/,
         string: /"(?:[^"\\]|\\.)*"/,
         number: /\b-?\d+\.?\d*\b/,
         keyword: /\b(?:true|false|null)\b/,
         fn: /(?!)x/,
       },
       bash: {
         comment: /#.*/,
         string: /("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/,
         number: /\b\d+\b/,
         keyword: /\b(?:if|then|else|elif|fi|for|while|do|done|case|esac|function|return|export|local|echo|exit|in|set|source)\b|(?:^|\s)(?:-{1,2}[A-Za-z][\w-]*)/,
         fn: /\b(?:pip|python3?|npm|node|git|curl|wget|sudo|apt|apt-get|systemctl|docker|cd|ls|mkdir|rm|cp|mv|chmod|chown)\b/,
       },
       sql: {
         comment: /--.*|\/\*[\s\S]*?\*\//,
         string: /'(?:[^'\\]|\\.)*'/,
         number: /\b\d+\.?\d*\b/,
         keyword: /\b(?:SELECT|FROM|WHERE|INSERT|INTO|VALUES|UPDATE|SET|DELETE|CREATE|TABLE|ALTER|DROP|JOIN|LEFT|RIGHT|INNER|OUTER|ON|AS|AND|OR|NOT|NULL|IS|IN|LIKE|GROUP|BY|ORDER|HAVING|LIMIT|DISTINCT|UNION|ALL|EXISTS|CASE|WHEN|THEN|END|PRIMARY|KEY|FOREIGN|REFERENCES|DEFAULT)\b/i,
         fn: /\b([A-Za-z_]\w*)(?=\()/,
       },
       yaml: {
         comment: /#.*/,
         string: /("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/,
         number: /\b\d+\.?\d*\b/,
         keyword: /\b(?:true|false|null|yes|no)\b/i,
         fn: /(?!)x/,
       },
       html: {
         comment: /<!--[\s\S]*?-->/,
         string: /("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/,
         number: /(?!)x/,
         keyword: /<\/?[a-zA-Z][\w-]*|\/?>/,
         fn: /(?!)x/,
       },
       css: {
         comment: /\/\*[\s\S]*?\*\//,
         string: /("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/,
         number: /\b\d+\.?\d*(?:px|em|rem|%|s|ms|vh|vw)?\b/,
         keyword: /[.#]?[a-zA-Z-]+(?=\s*[:{])/,
         fn: /(?!)x/,
       },
};
HL_RULES.py = HL_RULES.python;
HL_RULES.js = HL_RULES.javascript; HL_RULES.jsx = HL_RULES.javascript;
HL_RULES.ts = HL_RULES.javascript; HL_RULES.typescript = HL_RULES.javascript; HL_RULES.tsx = HL_RULES.javascript;
HL_RULES.sh = HL_RULES.bash; HL_RULES.shell = HL_RULES.bash; HL_RULES.zsh = HL_RULES.bash;
HL_RULES.yml = HL_RULES.yaml;
function stripGroups(src) {
       return src.replace(/\((?!\?)/g, '(?:');
}
function highlight(code, lang) {
       const rules = HL_RULES[(lang||'').toLowerCase()];
       // Generic fallback tokenizer for unknown languages
       if (!rules) {
           const fallbackRules = {
               comment: /\/\/.*|\#.*|\/\*[\s\S]*?\*\//,
               string: /("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/,
               number: /\b-?\d+\.?\d*\b/,
               keyword: /(?!)x/,
               fn: /(?!)x/,
           };
           const combined = new RegExp(
               `(?<C>${stripGroups(fallbackRules.comment.source)})|` +
               `(?<S>${stripGroups(fallbackRules.string.source)})|` +
               `(?<N>${stripGroups(fallbackRules.number.source)})`,
               'g'
           );
           let out = '', last = 0, m;
           while ((m = combined.exec(code)) !== null) {
               out += esc(code.slice(last, m.index));
               const g = m.groups || {};
               if      (g.C !== undefined) out += '<span class="tok-c">'+esc(g.C)+'</span>';
               else if (g.S !== undefined) out += '<span class="tok-s">'+esc(g.S)+'</span>';
               else if (g.N !== undefined) out += '<span class="tok-n">'+esc(g.N)+'</span>';
               else out += esc(m[0]);
               last = combined.lastIndex;
               if (m.index === combined.lastIndex) combined.lastIndex++;
           }
           out += esc(code.slice(last));
           return out;
       }
       const hasI = [rules.comment, rules.string, rules.keyword].some(r => r.flags.includes('i'));
       const combined = new RegExp(
         `(?<C>${stripGroups(rules.comment.source)})|(?<S>${stripGroups(rules.string.source)})|`+
         `(?<N>${stripGroups(rules.number.source)})|(?<K>${stripGroups(rules.keyword.source)})|`+
         `(?<F>${stripGroups(rules.fn.source)})`,
         'g' + (hasI ? 'i' : '')
       );
       let out = '', last = 0, m;
       while ((m = combined.exec(code)) !== null) {
         out += esc(code.slice(last, m.index));
         const g = m.groups || {};
         if      (g.C !== undefined) out += '<span class="tok-c">'+esc(g.C)+'</span>';
         else if (g.S !== undefined) out += '<span class="tok-s">'+esc(g.S)+'</span>';
         else if (g.N !== undefined) out += '<span class="tok-n">'+esc(g.N)+'</span>';
         else if (g.K !== undefined) out += '<span class="tok-k">'+esc(g.K)+'</span>';
         else if (g.F !== undefined) out += '<span class="tok-f">'+esc(g.F)+'</span>';
         else out += esc(m[0]);
         last = combined.lastIndex;
         if (m.index === combined.lastIndex) combined.lastIndex++;
       }
       out += esc(code.slice(last));
       return out;
}
function langToExt(lang) {
       const map = { python:'py', py:'py', javascript:'js', js:'js', jsx:'jsx', typescript:'ts', ts:'ts',
         tsx:'tsx', bash:'sh', sh:'sh', shell:'sh', zsh:'sh', json:'json', sql:'sql', yaml:'yaml', yml:'yaml',
         html:'html', css:'css', markdown:'md', md:'md', c:'c', cpp:'cpp', 'c++':'cpp', java:'java',
         go:'go', rust:'rs', rb:'rb', ruby:'ruby', php:'php', xml:'xml', text:'txt', plaintext:'txt' };
       return map[(lang||'').toLowerCase()] || 'txt';
}
function md(text) {
       if (!text) return '';
       let s = sanitizeAssistantResponse(String(text));
       const codeBlocks = [];
       s = s.replace(/```([^\n]*)\n([\s\S]*?)```/g, (_, lang, code) => {
         const idx = codeBlocks.length;
         codeBlocks.push({ lang: (lang||'').trim(), code: code.replace(/\n$/,'') });
         return '\x00' + idx + '\x01';
       });
       const inlineCode = [];
       s = s.replace(/`([^`\n]+)`/g, (_, code) => {
         const idx = inlineCode.length;
         inlineCode.push(code);
         return '\x00' + idx + '\x02';
       });
       s = esc(s);
       s = s.replace(/^######\s+(.+)$/gm,'<h6>$1</h6>');
       s = s.replace(/^#####\s+(.+)$/gm, '<h5>$1</h5>');
       s = s.replace(/^####\s+(.+)$/gm,  '<h4>$1</h4>');
       s = s.replace(/^###\s+(.+)$/gm,   '<h3>$1</h3>');
       s = s.replace(/^##\s+(.+)$/gm,    '<h2>$1</h2>');
       s = s.replace(/^#\s+(.+)$/gm,     '<h1>$1</h1>');
       s = s.replace(/^---+$/gm, '<hr>');
       s = s.replace(/^&gt;\s?(.+)$/gm, '<blockquote>$1</blockquote>');
       s = s.replace(/^(\|.+\|\s*\n\|[-:| ]+\|\s*\n(?:\|.+\|\s*\n?)+)/gm, t => {
         const rows = t.trim().split('\n').filter(r => r.trim());
         if (rows.length < 2) return t;
         const head = rows[0];
         const body = rows.slice(2);
         const ths = head.split('|').slice(1,-1).map(c => '<th>' + esc(c.trim()) + '</th>').join('');
         if (!ths) return t;
         const trs = body.map(r => {
           const cells = r.split('|').slice(1,-1).map(c => '<td>' + esc(c.trim()) + '</td>').join('');
           return '<tr>' + cells + '</tr>';
         }).join('');
         return '<table class="markdown-table"><thead><tr>' + ths + '</tr></thead><tbody>' + trs + '</tbody></table>';
       });
       s = s.replace(/(?:^|\n)([-*+]\s+.+(?:\n(?:[-*+]\s+.+|[ \t]+.+))*)/gm, m => {
         const lines = m.split('\n').filter(l => l.trim());
         const items = lines.map(l => '<li>' + l.replace(/^[-*+]\s+/, '') + '</li>').join('');
         return '\n<ul>'+items+'</ul>';
       });
       s = s.replace(/(?:^|\n)((?:\d+\.\s+.+(?:\n(?:\d+\.\s+.+|[ \t]+.+))*)|(?:\d+\.\s+.+))/gm, m => {
         const lines = m.split('\n').filter(l => l.trim());
         const items = lines.map(l => '<li>' + l.replace(/^\d+\.\s+/, '') + '</li>').join('');
         return '\n<ol>'+items+'</ol>';
       });
       s = s.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
       s = s.replace(/\*\*(.+?)\*\*/g,     '<strong>$1</strong>');
       s = s.replace(/__(.+?)__/g,         '<strong>$1</strong>');
       s = s.replace(/\*([^*\n]+)\*/g,      '<em>$1</em>');
       s = s.replace(/_([^_\n]+)_/g,        '<em>$1</em>');
       s = s.replace(/~~(.+?)~~/g,          '<del>$1</del>');
       s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
       s = s.replace(/\n{2,}/g, '\n');
       const parts = s.split('\n');
       s = parts.map(p => {
         p = p.trim();
         if (!p) return '';
         if (/^<(h[1-6]|hr|ul|ol|table|blockquote)/.test(p)) return p;
         if (/^\x00\d+\x01$/.test(p)) return p;
         return '<p>' + p.replace(/\n/g, '<br>') + '</p>';
       }).join('\n');
       s = s.replace(/\x00(\d+)\x02/g, (_, idx) =>
         '<code class="inline-code">' + esc(inlineCode[parseInt(idx)]) + '</code>');
       s = s.replace(/\x00(\d+)\x01/g, (_, idx) => {
         const {lang, code} = codeBlocks[parseInt(idx)];
         const displayLang = lang || 'text';
         const id = 'cb' + Math.random().toString(36).slice(2,8);
         const highlighted = highlight(code, lang);
         return '<pre class="code-block"><div class="pre-head">'+
'<span class="pre-lang">'+esc(displayLang)+'</span>'+
'<div class="pre-actions">'+
'<button class="pre-btn" onclick="copyCode(this,\''+id+'\')" id="'+id+'" title="Copy code">'+
'<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg>'+
'<span>Copy</span></button>'+
'<button class="pre-btn" onclick="downloadCode(\''+id+'\',\''+esc(displayLang)+'\')" title="Download as file">'+
'<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/></svg>'+
'<span>Download</span></button>'+
'</div></div>'+
'<code class="hl-code" data-lang="'+esc(displayLang)+'" id="data-'+id+'" data-raw="'+esc(code)+'">'+highlighted+'</code></pre>';
       });
       s = s.replace(/<p>\s*<\/p>/g,'');
       return s;
}

// ════════════════════════════════════════════════════════════ MESSAGE VALIDATION
function validateMessageContent(content, role) {
       if (!content) return null;
       const trimmed = String(content).trim();
       if (!trimmed || trimmed.length === 0) return null;
       if (/^[\s.,!?;:\-—–—""''""*#@]+$/.test(trimmed)) return null;
       return trimmed;
}
function escapeHtml(text) {
       const div = document.createElement('div');
       div.textContent = text;
       return div.innerHTML;
}

// ════════════════════════════════════════════════════════════
// UNIFIED BUBBLE DOM CONTRACT
// Every assistant/user bubble MUST be: .bub.<role> > .mc + .acts.
// All code paths (appendMsg, streaming, resumeTurn, error cards) go
// through buildBubble. That is what makes copyMsg/resumeTurn/retryTurn
// work uniformly across all message shapes.
// ════════════════════════════════════════════════════════════
function buildBubble(role, turnId) {
    const row = document.createElement('div');
    row.className = 'row ' + role;
    if (turnId) row.setAttribute('data-turn-id', turnId);
    const bub = document.createElement('div');
    bub.className = 'bub ' + role;
    const mc = document.createElement('div');
    mc.className = 'mc';
    bub.appendChild(mc);
    const acts = document.createElement('div');
    acts.className = 'acts';
    bub.appendChild(acts);
    row.appendChild(bub);
    return { row, bub, mc, acts };
}

function makeCopyRetryActions(turnId) {
    return (
        '<button class="act" onclick="copyMsg(this)" title="Copy">' +
        '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg>Copy</button>' +
        '<button class="act retry" onclick="retryTurn(\'' + (turnId || '') + '\')" title="Retry this response">' +
        '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M17.65 6.35A7.958 7.958 0 0012 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08A5.99 5.99 0 0112 18c-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/></svg>Retry</button>'
    );
}
function makeCopyRetryContinueActions(turnId) {
    return makeCopyRetryActions(turnId) +
        '<button class="act continue" onclick="resumeTurn(\'' + (turnId || '') + '\')" title="Continue from interruption">' +
        '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>Continue</button>';
}

function appendErrorCard(errorMsg, title = '❌ Error', turnId = null) {
       const chat = document.getElementById('chat');
       if (!chat) return;
       const { row, bub, mc, acts } = buildBubble('assistant', turnId);
       bub.style.borderColor = '#ef4444';
       bub.style.background = '#2a1a1a';
       bub.classList.add('error-card');
       mc.innerHTML = `<strong>${escapeHtml(title)}</strong><br/>${escapeHtml(errorMsg)}`;
       if (turnId) {
           acts.innerHTML = (
               '<button class="act retry" onclick="retryTurn(\'' + turnId + '\')" title="Retry this response">' +
               '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M17.65 6.35A7.958 7.958 0 0012 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08A5.99 5.99 0 0112 18c-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/></svg>Retry</button>'
           );
       }
       chat.appendChild(row);
       chat.scrollTop = chat.scrollHeight;
       log.error(`Error card shown: ${errorMsg}`);
}

function appendMsg(role, content, animate=true, turnId=null) {
       const chat = document.getElementById('chat');
       if (!chat) return;
       const validatedContent = validateMessageContent(content, role);
       if (!validatedContent && role === 'assistant' && !animate) {
         appendErrorCard('Assistant returned an empty response. Please try again.', '❌ Error', turnId);
         return;
       }
       const displayContent = validatedContent || (animate ? '' : '[No response]');
       const empty = document.getElementById('empty');
       if (empty) empty.remove();

       const { row, bub, mc, acts } = buildBubble(role, turnId);
       // Always stash the raw markdown/text on the .mc element so copyMsg()
       // can recover it even after a page reload (when turnRequests is empty
       // but the DOM is rehydrated by loadConv).
       mc.dataset.raw = displayContent;
       if (role === 'user') {
         mc.innerHTML = esc(displayContent).replace(/\n/g,'<br>');
       } else {
         mc.innerHTML = md(displayContent);
       }
       if (role === 'assistant' && !animate) {
         const isInterrupted = content && (
             content.includes('[Response interrupted') ||
             content.includes('⚡INTERRUPTED') ||
             content.includes('⚡TRUNCATED') ||
             content.includes('⚡STALLED')
         );
         if (isInterrupted) {
           bub.classList.add('interrupted');
           acts.innerHTML = makeCopyRetryContinueActions(turnId);
         } else {
           acts.innerHTML = makeCopyRetryActions(turnId);
         }
       } else if (role === 'user') {
         acts.innerHTML = (
           '<button class="act" onclick="copyMsg(this)" title="Copy">' +
           '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg>Copy</button>' +
           '<button class="act retry" onclick="retryUserTurn(\'' + (turnId || '') + '\')" title="Retry this request">' +
           '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M17.65 6.35A7.958 7.958 0 0012 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08A5.99 5.99 0 0112 18c-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/></svg>Retry</button>'
         );
       }
       chat.appendChild(row);
       if (animate) chat.scrollTop = chat.scrollHeight;
       return mc;
}

// ════════════════════════════════════════════════════════════ CLIPBOARD
function copyToClipboard(text) {
    if (navigator.clipboard && window.isSecureContext) {
        return navigator.clipboard.writeText(text);
    }
    return new Promise((resolve, reject) => {
        try {
            const ta = document.createElement('textarea');
            ta.value = text;
            ta.setAttribute('readonly', '');
            ta.style.position = 'fixed';
            ta.style.top = '0';
            ta.style.left = '-9999px';
            // Append inside focused container so focus-trapped modals don't
            // steal selection and make execCommand('copy') fail.
            const container =
                document.querySelector('#settings-modal.show')
                || document.activeElement?.closest?.('#settings-modal.show')
                || document.body;
            container.appendChild(ta);
            ta.focus();
            ta.select();
            ta.setSelectionRange(0, text.length);
            const ok = document.execCommand('copy');
            container.removeChild(ta);
            if (ok) resolve();
            else reject(new Error('execCommand returned false'));
        } catch (e) {
            reject(e);
        }
    });
}

// Prefer raw markdown (from live stream state OR the data-raw attribute
// stashed on .mc). Fall back to textContent so old messages still copy
// *something* rather than silently failing.
function copyMsg(btn) {
       const row = btn.closest('.row');
       const turnId = row ? row.getAttribute('data-turn-id') : null;
       const isAssistant = row && row.classList.contains('assistant');
       let text = null;

       // 1. Live-stream state — ONLY for assistant rows; it holds the model's
       //    reply, not the user's prompt.
       if (isAssistant && turnId && turnRequests[turnId] && turnRequests[turnId].rawMarkdown) {
           text = turnRequests[turnId].rawMarkdown;
       }
       // 2. data-raw on the .mc (per-bubble, set by appendMsg/stream/resume).
       if (!text && row) {
           const mcEl = row.querySelector('.mc');
           if (mcEl && mcEl.dataset && mcEl.dataset.raw) {
               text = mcEl.dataset.raw;
           }
       }
       // 3. Last resort — rendered text.
       if (!text && row) {
           const mcEl = row.querySelector('.mc');
           text = mcEl ? (mcEl.textContent || '') : '';
       }
       if (!text) {
           toast('Nothing to copy', 'warning');
           return;
       }
       copyToClipboard(text).then(() => {
         btn.classList.add('ok');
         setTimeout(() => btn.classList.remove('ok'), 2000);
       }).catch(err => {
         toast('❌ Copy failed: ' + (err.message || 'unknown'), 'error');
       });
}

function copyCode(btn, id) {
       const el = document.getElementById('data-' + id);
       const code = el ? (el.getAttribute('data-raw') ?? el.textContent) : '';
       if (!code) {
           toast('Nothing to copy', 'warning');
           return;
       }
       copyToClipboard(code).then(() => {
         btn.classList.add('ok');
         const label = btn.querySelector('span');
         if (label) label.textContent = 'Copied!';
         setTimeout(() => {
           btn.classList.remove('ok');
           if (label) label.textContent = 'Copy';
         }, 2000);
       }).catch(err => {
         toast('❌ Copy failed: ' + (err.message || 'unknown'), 'error');
       });
}
function downloadCode(id, lang) {
       const el = document.getElementById('data-' + id);
       const code = el ? (el.getAttribute('data-raw') ?? el.textContent) : '';
       if (!code) return;
       const ext = langToExt(lang);
       const blob = new Blob([code], { type: 'text/plain;charset=utf-8' });
       const a = document.createElement('a');
       a.href = URL.createObjectURL(blob);
       a.download = `snippet-${id}.${ext}`;
       document.body.appendChild(a);
       a.click();
       a.remove();
       setTimeout(() => URL.revokeObjectURL(a.href), 1000);
       toast(`✅ Downloaded snippet-${id}.${ext}`, 'success', 1800);
}

// ════════════════════════════════════════════════════════════
// RESUME (CONTINUE) FUNCTION
//
// Server contract: POST /v1/chat/resume returns text/event-stream, where
// each frame is "data: {json}\n\n" with a chat-completion chunk, terminated
// by "data: [DONE]\n\n". This is the ONLY streaming endpoint on the server
// that emits SSE — /v1/chat/stream and /v1/chat/smart emit raw text — so
// resumeTurn must decode SSE frames before feeding tokens to the parser.
//
// Other fixes vs. the previous implementation:
//   * uses the global `ctrl` so Stop / Esc / beforeunload can abort us;
//   * uses an IDLE timer (reset on every chunk) instead of a total timer;
//   * reconstructs state from the DOM if turnRequests[turnId] was lost to a
//     page reload — the bubble's .mc data-raw and the sibling user row are
//     enough to rebuild turnIndex + partial markdown;
//   * reads X-Prompt-Tokens so the context bar stays honest;
//   * handles 403 / 413 / 429 with specific toasts;
//   * max_tokens is now 0 = unlimited so the server resumes until EOS.
//     The old RESUME_CHUNK cap forced a Continue click every ~1280 tokens.
//   * recognises both canonical ⚡ markers and the legacy
//     "[Response interrupted …]" text the engine can still emit.
//   * auto-resumes on TRUNCATED / STALLED so the user does not have to
//     click Continue for a server-side boundary they never requested.
// ════════════════════════════════════════════════════════════
let resumeInProgress = false;
async function resumeTurn(turnId) {
       if (busy || resumeInProgress || !online) {
         toast('Cannot resume – server offline or busy', 'warning');
         return;
       }
       const turnEl = document.querySelector(`.row.assistant[data-turn-id="${turnId}"]`);
       if (!turnEl) { toast('No message to resume', 'error'); return; }
       const mc = turnEl.querySelector('.mc');
       if (!mc) { toast('Message body not found', 'error'); return; }

       const convId = cid;
       if (!convId) { toast('No conversation ID', 'error'); return; }

       // ── Reconstruct state if it isn't in memory (page reload, tab swap) ──
       let state = turnRequests[turnId];
       if (!state || !state.interruption) {
         const conv = curConv();
         // Locate the preceding user row in the DOM and count user rows up to
         // and including it — that gives us the matching index into
         // conv.msgs, since chat history is strictly append-only.
         let userIdx = -1;
         try {
           const chatEl = document.getElementById('chat');
           const allRows = Array.from(chatEl.querySelectorAll('.row'));
           const myIdx = allRows.indexOf(turnEl);
           let userRowsSeen = 0;
           for (let i = 0; i <= myIdx && i < allRows.length; i++) {
             if (allRows[i].classList.contains('user')) userRowsSeen++;
           }
           if (conv && userRowsSeen > 0) {
             let seen = 0;
             for (let i = 0; i < conv.msgs.length; i++) {
               if (conv.msgs[i].role === 'user') {
                 seen++;
                 if (seen === userRowsSeen) { userIdx = i; break; }
               }
             }
           }
         } catch (_) { /* best-effort */ }
         const rawNow = mc.dataset.raw || mc.textContent || '';
         state = {
           turnIndex: userIdx,
           rawMarkdown: rawNow,
           interruption: { kind: 'INTERRUPTED', tokens: 0 },
           params: {
             modelId: currentModelId,
             temperature: currentTemperature,
             maxTokens: null,
           },
         };
         turnRequests[turnId] = state;
         log.info('resumeTurn: reconstructed missing state for', turnId);
       }

       // FIX-4a: prefer raw markdown over rendered textContent.
       const cleanContent = stripStreamMarker(mc.dataset.raw || mc.textContent || '');
       // Limits come from /v1/runtime/limits — mostly used here for the
       // context bar. max_tokens is sent as 0 (unlimited) so the server
       // resumes until EOS; the engine slides its own window if needed.
       const ctxLimit = Number(runtimeLimits.context_length) || modelContextLength || 2048;
       // 0 = unlimited. The server resumes until EOS / stop / repetition
       // guard; it slides its own context window if needed. The old
       // RESUME_CHUNK cap forced a Continue click every ~1280 tokens.
       const remaining = 0;

       const acts = turnEl.querySelector('.acts');
       if (acts) {
         const contBtn = acts.querySelector('.act.continue');
         if (contBtn) contBtn.remove();
         const retryBtn = acts.querySelector('.act.retry');
         if (retryBtn) retryBtn.disabled = true;
       }
       mc.dataset.raw = cleanContent;
       mc.innerHTML = md(cleanContent) || esc(cleanContent);
       if (cleanContent && !mc.innerHTML.endsWith(' ')) {
         mc.innerHTML += ' ';
       }
       const cursorSpan = document.createElement('span');
       cursorSpan.className = 'cursor';
       mc.appendChild(cursorSpan);

       // ── Set the global controller so Stop/Esc/beforeunload can abort us ──
       if (ctrl) { try { ctrl.abort(); } catch (_) {} }
       ctrl = new AbortController();
       const controller = ctrl;
       currentStreamAborted = false;

       resumeInProgress = true;
       busy = true;
       updateUIStates();
       showThinkingIndicator();

       const timeoutVal = getTimeoutSeconds();
       let timeoutId = null;
       const resetStreamIdleTimer = () => {
         if (timeoutId) clearTimeout(timeoutId);
         if (timeoutVal > 0) {
           timeoutId = setTimeout(() => {
             if (ctrl) { try { ctrl.abort(); } catch (_) {} }
           }, timeoutVal * 1000);
         }
       };
       resetStreamIdleTimer();

       try {
         const body = {
           conv_id: convId,
           max_tokens: remaining,
           stream: true,
           temperature: currentTemperature,
           // Forwarded for the server's benefit if its ResumeRequest schema
           // is extended to accept them; harmless if ignored (Pydantic
           // default is to drop unknown fields silently).
           model: state.params?.modelId || currentModelId || undefined,
           no_think: !thinkingModeEnabled,
           history_mode: 'replace',
         };
         const headers = buildRequestHeaders();  // includes X-Session-ID
         const resp = await fetch(BASE + '/v1/chat/resume', {
           method: 'POST',
           headers: headers,
           body: JSON.stringify(body),
           signal: controller.signal
         });

         // Context bar: prefer server-authoritative prompt-token count.
         const serverPromptTokens = parseInt(
           resp.headers.get('X-Prompt-Tokens') || '0', 10
         );
         if (serverPromptTokens > 0) {
           updateTokenIndicator(serverPromptTokens, ctxLimit);
         }

         if (!resp.ok) {
           let errBody = '';
           try { errBody = await resp.text(); } catch (_) {}
           if (resp.status === 404) {
             toast('No checkpoint found for this conversation', 'error');
           } else if (resp.status === 403) {
             toast('Invalid API key. Update it in settings (⚙ Key).', 'error');
           } else if (resp.status === 413) {
             toast(`Prompt exceeds context: ${errBody.slice(0, 200)}`, 'error', 6000);
           } else if (resp.status === 429) {
             toast('Rate limited — please wait before resuming', 'warning', 4000);
           } else if (resp.status === 503) {
             toast('Model not ready – please wait', 'warning');
           } else {
             toast(`Resume failed (HTTP ${resp.status}): ${errBody.slice(0, 200)}`,
                   'error', 6000);
           }
           cursorSpan.remove();
           if (acts) {
             const retryBtn = acts.querySelector('.act.retry');
             if (retryBtn) retryBtn.disabled = false;
           }
           return;
         }

         const reader = resp.body.getReader();
         const decoder = new TextDecoder();
         let fullText = cleanContent;
         let hasContent = false;
         const parser = new RobustStreamParser();
         let resumeMarker = null;
         let sseBuf = '';

         const paintNow = () => {
           mc.dataset.raw = fullText;
           mc.innerHTML = md(fullText) || esc(fullText);
           mc.appendChild(cursorSpan);
           const chatEl = document.getElementById('chat');
           if (chatEl) chatEl.scrollTop = chatEl.scrollHeight;
         };

         const consumeToken = (text) => {
           if (!text) return;
           // ⚡ markers may be inlined into an SSE delta if the engine (or a
           // wrapper) ever emits them directly.
           const rmark = parseStreamMarker(text);
           if (rmark) { resumeMarker = rmark; return; }
           // Legacy text markers the engine can still emit on cancellation —
           // the server's own startswith() check misses them because the
           // payload begins with two newlines, so we catch them client-side.
           if (/^\s*\[(Response interrupted|Generation paused)[^\]]*\]\s*$/.test(text)) {
             resumeMarker = resumeMarker || { kind: 'INTERRUPTED', tokens: 0 };
             return;
           }
           if (/^\s*\[Error:[^\]]*\]\s*$/.test(text)) {
             resumeMarker = resumeMarker || { kind: 'STALLED', tokens: 0 };
             return;
           }
           const filtered = filterThinkingAndMetaText(text);
           if (filtered) {
             fullText += filtered;
             hasContent = true;
             paintNow();
           }
         };

         // ── SSE frame loop ─────────────────────────────────────────────
         // Frames are separated by "\n\n". Each carries one or more "data:"
         // lines that we concatenate per the SSE spec, then JSON.parse.
         while (true) {
           let chunk;
           try {
             chunk = await reader.read();
           } catch (e) {
             if (controller.signal.aborted) break;
             throw e;
           }
           if (chunk.done) break;
           resetStreamIdleTimer();

           sseBuf += decoder.decode(chunk.value, { stream: true });

           let sep;
           while ((sep = sseBuf.indexOf('\n\n')) !== -1) {
             let rawFrame = sseBuf.slice(0, sep);
             sseBuf = sseBuf.slice(sep + 2);
             // Tolerate CRLF at frame boundaries.
             rawFrame = rawFrame.replace(/\r/g, '');

             const dataLines = [];
             for (const line of rawFrame.split('\n')) {
               if (line.startsWith('data:')) {
                 dataLines.push(line.slice(5).replace(/^ /, ''));
               }
             }
             if (!dataLines.length) continue;
             const data = dataLines.join('\n');

             if (data === '[DONE]') { continue; }

             let obj;
             try { obj = JSON.parse(data); }
             catch (_) { continue; }

             // Server can frame {"error": "..."} on mid-stream failures.
             if (obj && typeof obj === 'object' && obj.error) {
               log.warn('Resume stream error frame:', obj.error);
               continue;
             }

             const choice = obj?.choices?.[0];
             if (!choice) continue;

             // finish_reason === "interrupted" is the server's way of saying
             // it detected the canonical interruption payload and gave up.
             if (choice.finish_reason === 'interrupted') {
               resumeMarker = resumeMarker || { kind: 'INTERRUPTED', tokens: 0 };
             }

             const delta = choice.delta?.content ?? '';
             if (delta) {
               for (const token of parser.push(delta)) consumeToken(token);
             }
           }
         }

         // Drain the parser (partial token, unclosed thinking block, etc.).
         for (const token of parser.flush()) consumeToken(token);
         cursorSpan.remove();

         // ── Post-stream handling ──────────────────────────────────────
         if (hasContent && fullText) {
           // Update conv.msgs by turn index, not string search.
           const conv = curConv();
           if (conv && typeof state.turnIndex === 'number' && state.turnIndex >= 0) {
             if (conv.msgs[state.turnIndex + 1]) {
               conv.msgs[state.turnIndex + 1].content = fullText;
             } else {
               conv.msgs.push({ role: 'assistant', content: fullText });
             }
             save();
           }
           // Normalize bubble state.
           const bub = turnEl.querySelector('.bub.assistant');
           if (bub) bub.classList.remove('interrupted');
           if (acts) {
             acts.innerHTML = resumeMarker
               ? makeCopyRetryContinueActions(turnId)
               : makeCopyRetryActions(turnId);
           }
           state.rawMarkdown = fullText;
           state.interruption = resumeMarker
             ? { kind: resumeMarker.kind, tokens: resumeMarker.tokens }
             : null;

           if (resumeMarker) {
             // Same auto-resume policy as streamReply: TRUNCATED/STALLED
             // are server-side boundaries, not user-requested stops, so
             // keep going without making the user click.
             window.__autoResume = window.__autoResume || {};
             const _arN = (window.__autoResume[turnId] =
                           (window.__autoResume[turnId] || 0) + 1);
             const _willAuto = (
               (resumeMarker.kind === 'TRUNCATED' || resumeMarker.kind === 'STALLED')
               && !currentStreamAborted
               && _arN <= 3
             );
             if (_willAuto) {
               toast(`⚠️ Still ${resumeMarker.kind.toLowerCase()} — auto-continuing…`,
                     'warning', 2500);
               setTimeout(() => resumeTurn(turnId), 400);
             } else {
               toast('⚠️ Still cut off — click Continue again', 'warning', 2500);
             }
           } else {
             toast('✅ Resumed successfully', 'success', 2500);
           }
         } else {
           // No new content at all — leave the Continue button in place so the
           // user can try again without losing the partial.
           if (acts && !acts.querySelector('.act.continue')) {
             acts.innerHTML = makeCopyRetryContinueActions(turnId);
           }
           toast('⚠️ No additional content received from resume', 'warning');
         }
       } catch (e) {
         try { cursorSpan.remove(); } catch (_) {}
         if (controller.signal.aborted) {
           if (currentStreamAborted) {
             toast('⏸ Resuming was stopped', 'info');
           } else if (timeoutVal > 0) {
             toast(`⏱ Resume timed out after ${timeoutVal}s — click Continue to try again`,
                   'warning', 5000);
           } else {
             toast('⏸ Resuming was stopped', 'info');
           }
         } else {
           log.error('Resume error:', e);
           toast('❌ Resume failed: ' + (e.message || 'unknown error'), 'error');
         }
       } finally {
         if (timeoutId) clearTimeout(timeoutId);
         resumeInProgress = false;
         busy = false;
         if (ctrl === controller) ctrl = null;
         updateUIStates();
         hideThinkingIndicator();
       }
}

// ══════════════════════════════════════ AUTO-RESIZE
function autoResize() {
       const ta = document.getElementById('prompt');
       ta.style.height = 'auto';
       ta.style.height = Math.min(ta.scrollHeight, 200) + 'px';
}
document.getElementById('prompt').addEventListener('input', autoResize);
document.getElementById('prompt').addEventListener('keydown', e => {
       if (e.key === 'Enter' && !e.shiftKey) {
         e.preventDefault();
         send();
       } else if (e.key === 'Escape') {
         stopGen();
       }
});
// ═══════════════════════════════════════ KEYBOARD SHORTCUTS
document.addEventListener('keydown', e => {
       if ((e.ctrlKey || e.metaKey) && e.key === 'e') {
         e.preventDefault();
         exportConv();
       }
       if ((e.ctrlKey || e.metaKey) && e.key === 'i') {
         e.preventDefault();
         importConv();
       }
});
// ════════════════════════════════════════════════════════════ SETTINGS MODAL
function showSettings() {
       ModalManager.open();
}
// Always send X-Session-ID so client cid == server session_id == checkpoint key.
function buildRequestHeaders() {
       const headers = { 'Content-Type': 'application/json' };
       if (apiKey) headers['X-API-Key'] = apiKey;
       if (cid) headers['X-Session-ID'] = cid;
       return headers;
}
function getTimeoutSeconds() {
       const sel = document.getElementById('timeout-sel');
       if (!sel) return 300;
       const value = parseInt(sel.value, 10);
       return value > 0 ? value : 0;
}
// NOTE: addTimeoutToRequest is intentionally NOT called from streamReply.
// The server interprets timeout_s as a HARD TOTAL deadline, but this dropdown
// is a client-side IDLE timer. F6 in the audit documents the mismatch.
// Kept for reference/other call sites; do not wire it into the chat send path.
function addTimeoutToRequest(body) {
       const timeoutSec = getTimeoutSeconds();
       if (timeoutSec > 0) body.timeout_s = timeoutSec;
       return body;
}

// ════════════════════════════════════════════════════════════
// Retry primitives
// buildMessages: construct the exact message array for a given history point.
// streamReply: shared streaming implementation used by send() and retry*().
// Both retries preserve turnId and NEVER push a duplicate user message.
// ════════════════════════════════════════════════════════════

function buildMessages(conv, upToIndex) {
    const out = [];
    const sys = document.getElementById('sys-in').value.trim();
    if (sys) out.push({ role: 'system', content: sys });
    const limit = Math.min(upToIndex, conv.msgs.length);
    for (let i = 0; i < limit; i++) out.push(conv.msgs[i]);
    return out;
}

async function retryTurn(turnId) {
    if (busy || !online) {
        toast('Cannot retry: server offline or busy', 'warning');
        return;
    }
    const state = turnRequests[turnId];
    if (!state) { toast('No retry data for this turn', 'error'); return; }
    const conv = curConv();
    if (!conv) return;

    // Drop the failed assistant message and everything after it; we rebuild
    // the conversation from the preserved prefix. No new user message.
    if (conv.msgs.length > state.turnIndex + 1) {
        conv.msgs.splice(state.turnIndex + 1);
    }
    // Remove the DOM bubble for the failed reply and any error cards.
    const oldBubble = document.querySelector(`.row.assistant[data-turn-id="${turnId}"]`);
    if (oldBubble) oldBubble.remove();
    document.querySelectorAll('.row.assistant.error-card').forEach(el => el.remove());

    state.interruption = null;
    const messages = buildMessages(conv, state.turnIndex + 1);
    await streamReply(turnId, messages, state.params, true);
}

async function retryUserTurn(turnId) {
    if (busy || !online) { toast('Cannot retry: busy or offline', 'warning'); return; }
    const state = turnRequests[turnId];
    if (!state) { toast('No retry data for this turn', 'error'); return; }
    const conv = curConv();
    if (!conv) return;
    if (conv.msgs.length > state.turnIndex + 1) conv.msgs.splice(state.turnIndex + 1);
    const oldBubble = document.querySelector(`.row.assistant[data-turn-id="${turnId}"]`);
    if (oldBubble) oldBubble.remove();
    document.querySelectorAll('.row.assistant.error-card').forEach(el => el.remove());
    const messages = buildMessages(conv, state.turnIndex + 1);
    state.interruption = null;
    await streamReply(turnId, messages, state.params, true);
}

// ════════════════════════════════════════════════════════════
// streamReply — shared streaming implementation
// ════════════════════════════════════════════════════════════
async function streamReply(turnId, messages, params, isRetry) {
    const sel = document.getElementById('model-sel').value;
    const smart = sel === '__smart__';
    const ep = smart ? '/v1/chat/smart' : '/v1/chat/stream';

    const readyModels = allModels.filter(m => m.status === 'ready');
    const userReadyModels = filterUserModels(readyModels);
    let modelId = params.modelId;
    if (smart) {
        if (userReadyModels.length > 0) {
            modelId = userReadyModels[0].id;
        } else {
            toast('No user‑ready models available', 'error');
            return;
        }
    }
    if (!modelId) modelId = currentModelId;

    // All sizing values come from the server (cached runtimeLimits) so the
    // client automatically adapts when the user swaps in a bigger GPU or a
    // smaller model. Fallbacks mirror the server defaults.
    const CTX_TOTAL    = Number(runtimeLimits.context_length) || modelContextLength || 2048;
    const SAFETY       = Number(runtimeLimits.context_safety_tokens) || 256;
    const MIN_RESPONSE = Number(runtimeLimits.min_response_tokens) || 64;
    const REC_RESPONSE = Number(runtimeLimits.recommended_response_tokens) || 1024;
    // 0 = unlimited. The server slides its own context window; the client
    // never imposes a response-token ceiling. An explicit positive
    // params.maxTokens (e.g. an advanced caller) is still honoured.
    const responseBudget = (typeof params.maxTokens === 'number' && params.maxTokens > 0)
        ? params.maxTokens
        : 0;

    if (ctrl) ctrl.abort();
    ctrl = new AbortController();
    currentStreamAborted = false;
    const timeoutVal = getTimeoutSeconds();
    let timeoutId = null;
    let idleTimedOut = false;
    const resetStreamIdleTimer = () => {
        if (timeoutId) clearTimeout(timeoutId);
        if (timeoutVal > 0) {
            timeoutId = setTimeout(() => {
                idleTimedOut = true;
                if (ctrl) ctrl.abort();
            }, timeoutVal * 1000);
        }
    };
    resetStreamIdleTimer();
    busy = true;
    updateUIStates();
    showThinkingIndicator();
    showWorkIndicator('Sending…');
    const frameFilter = new FrameFilter();
    const t0 = performance.now();
    let tok = 0, ttft = 0, full = '', first = true;
    let assistantRow = null;
    let bubElement = null;
    let mcElement = null;
    let parser = new RobustStreamParser();
    let loadRetryCount = 0;
    const MAX_LOAD_RETRIES = 3;
    // coalesce DOM work to at most one md()+layout per animation frame.
    // Previously md(full) + scrollTop ran for EVERY streamed word (O(n^2)).
    let renderRaf = 0;
    const paintNow = () => {
        renderRaf = 0;
        if (mcElement) {
            mcElement.dataset.raw = full;
            mcElement.innerHTML = md(full) || esc(full);
        }
        const chatEl = document.getElementById('chat');
        if (chatEl) chatEl.scrollTop = chatEl.scrollHeight;
    };
    const scheduleStreamRender = () => {
        if (renderRaf) return;
        renderRaf = (typeof requestAnimationFrame === 'function')
            ? requestAnimationFrame(paintNow)
            : setTimeout(paintNow, 50);
    };

    const executeRequest = async (retryAfter = 0) => {
        if (retryAfter > 0) {
            await new Promise(resolve => setTimeout(resolve, retryAfter * 1000));
        }
        try {
            const headers = buildRequestHeaders();
            const body = {
                messages: messages,
                max_tokens: responseBudget,
                temperature: params.temperature,
                stream: true,
                no_think: !thinkingModeEnabled,
                history_mode: "replace",
            };
            // the server's ChatRequest requires `model` on EVERY chat
            // route (incl. /v1/chat/smart). Always send it; Auto-Router sends
            // the first ready user model and the server may still re-route.
            if (!modelId) {
                appendErrorCard('No ready model selected. Pick a model and retry.', '❌ Error', turnId);
                return;
            }
            body.model = modelId;
            // FIX-6: do NOT send timeout_s. The server treats it as a HARD
            // TOTAL deadline, but this dropdown is a client-side IDLE timer.
            // The server falls back to inference.default_timeout_s.
            const r = await fetch(BASE + ep, {
                method: 'POST',
                headers: headers,
                body: JSON.stringify(body),
                signal: ctrl.signal
            });

            // Prefer server-authoritative prompt tokens when available.
            const serverPromptTokens = parseInt(r.headers.get('X-Prompt-Tokens') || '0', 10);
            if (serverPromptTokens > 0) {
                updateTokenIndicator(serverPromptTokens, CTX_TOTAL);
            }

            if (!r.ok) {
                if (r.status === 503) {
                    loadRetryCount++;
                    if (loadRetryCount > MAX_LOAD_RETRIES) {
                        appendErrorCard(
                            `❌ Model did not become ready after ${MAX_LOAD_RETRIES} retries. ` +
                            `This usually means the model failed to load (e.g. insufficient RAM) ` +
                            `or the model name is not registered in the server. ` +
                            `Check server logs for details.`,
                            '❌ Error', turnId
                        );
                        return;
                    }
                    const retryAfterHeader = r.headers.get('Retry-After');
                    let waitSec = 5;
                    if (retryAfterHeader && !isNaN(parseInt(retryAfterHeader))) {
                        waitSec = parseInt(retryAfterHeader);
                    }
                    await executeRequest(waitSec);
                    return;
                } else if (r.status === 403) {
                    appendErrorCard('Invalid API key. Please update in settings (⚙ Key).', '❌ Error', turnId);
                    return;
                } else if (r.status === 429) {
                    appendErrorCard('Rate limited. Please wait before sending another message.', '❌ Error', turnId);
                    return;
                } else if (r.status === 413) {
                    const errBody = await r.text().catch(() => '');
                    appendErrorCard(`Prompt exceeds context: ${errBody.slice(0, 300)}`, '❌ Error', turnId);
                    return;
                }
                const err = await r.text().catch(() => '');
                appendErrorCard(`Server error ${r.status}: ${err.slice(0,300)}`, '❌ Error', turnId);
                return;
            }

            // Build the assistant bubble via the unified builder.
            if (!assistantRow) {
                const built = buildBubble('assistant', turnId);
                assistantRow = built.row;
                bubElement = built.bub;
                mcElement = built.mc;
                bubElement.setAttribute('data-streaming', 'true');
                document.getElementById('chat').appendChild(assistantRow);
            }

            const reader = r.body.getReader();
            const decoder = new TextDecoder();
            let hasContent = false;
            let interrupted = false;
            let interruptTokens = 0;
            let interruptKind = "INTERRUPTED";

            while (true) {
                let chunk;
                try {
                    chunk = await reader.read();
                } catch (e) {
                    if (ctrl.signal.aborted) break;
                    throw e;
                }
                if (chunk.done) break;
                resetStreamIdleTimer();
                const raw = decoder.decode(chunk.value, { stream: true });
                const ff = frameFilter.push(raw);
                for (const fr of ff.frames) updateWorkIndicator(fr);
                const s = ff.text;
                if (!s) continue;
                for (const token of parser.push(s)) { 
                    const marker = parseStreamMarker(token);
                    if (marker) {
                        interrupted = true;
                        interruptTokens = marker.tokens;
                        interruptKind = marker.kind;
                        continue;
                    }
                    const filtered = filterThinkingAndMetaText(token);
                    if (filtered) {
                        if (first) {
                            ttft = Math.round(performance.now() - t0);
                            first = false;
                            hideWorkIndicator();
                        }
                        full += filtered;
                        hasContent = true;
                        tok += Math.max(1, Math.round(filtered.length / 3.0));
                        scheduleStreamRender();
                        const ela_s = (performance.now() - t0) / 1000;
                        const instantTps = tok / Math.max(ela_s, 0.01);
                        showInstantTps(instantTps);
                        document.getElementById('st-ttft').textContent = ttft + 'ms';
                        document.getElementById('st-tok').textContent = tok;
                        document.getElementById('st-ela').textContent = ela_s.toFixed(1) + 's';
                    }
                }
            }
            for (const token of parser.flush()) {
                const marker = parseStreamMarker(token);
                if (marker) {
                    interrupted = true;
                    interruptTokens = marker.tokens;
                    interruptKind = marker.kind;
                    continue;
                }
                const filtered = filterThinkingAndMetaText(token);
                if (filtered) {
                    if (first) { ttft = Math.round(performance.now() - t0); first = false; }
                    full += filtered;
                    hasContent = true;
                    tok += Math.max(1, Math.round(filtered.length / 3.0));
                    if (mcElement) {
                        mcElement.dataset.raw = full;
                        mcElement.innerHTML = md(full) || esc(full);
                    }
                    const chatEl = document.getElementById('chat');
                    if (chatEl) chatEl.scrollTop = chatEl.scrollHeight;
                }
            }
            hideInstantTps();
            paintNow();   // FIX-5: guarantee the final text is painted before any end-state branch

            // ═══ Client-side abort path ═══
            if (ctrl.signal.aborted) {
                if (bubElement) {
                    bubElement.removeAttribute('data-streaming');
                    bubElement.classList.add('interrupted');
                        if (!full || !hasContent) {
                        full = currentStreamAborted
                            ? '⏸ Stopped by user'
                            : (idleTimedOut
                                ? `⏱ No data from the server for ${timeoutVal}s. The model may still be reading a long prompt — press Retry, or raise the timeout.`
                                : '⚠️ Connection closed before any content arrived. Press Retry.');
                        bubElement.classList.add('error-card');
                        if (mcElement) mcElement.textContent = full;
                        if (!currentStreamAborted) {
                            const acts0 = bubElement.querySelector('.acts') || document.createElement('div');
                            acts0.className = 'acts';
                            acts0.innerHTML = makeCopyRetryContinueActions(turnId);
                            bubElement.appendChild(acts0);
                        }
                    } else {
                        const note = document.createElement('div');
                        note.style.color = '#f59e0b';
                        note.style.fontSize = '12px';
                        note.style.marginTop = '8px';
                        note.textContent = '⏸ Generation stopped (partial content) — click Continue to resume from checkpoint';
                        bubElement.appendChild(note);
                        const acts = bubElement.querySelector('.acts') || document.createElement('div');
                        acts.className = 'acts';
                        acts.innerHTML = makeCopyRetryContinueActions(turnId);
                        bubElement.appendChild(acts);
                    }
                }
                if (full && full !== '⏸ Stopped by user' && hasContent) {
                    turnRequests[turnId] = turnRequests[turnId] || {};
                    turnRequests[turnId].rawMarkdown = full;
                    turnRequests[turnId].interruption = { kind: 'INTERRUPTED', tokens: tok };
                    const convObj = curConv();
                    if (convObj && typeof params.turnIndex === 'number') {
                        convObj.msgs.push({ role: 'assistant', content: full });
                        save();
                    }
                }
                if (timeoutId && timeoutVal > 0 && !currentStreamAborted) {
                    toast(`⏱ Request timed out after ${timeoutVal}s. Partial content preserved. Click Continue to resume.`, 'warning', 6000);
                }
                return;
            }

            // ═══ Interrupted / truncated / stalled path ═══
            if (interrupted) {
                const cleanFull = stripStreamMarker(full);
                if (cleanFull || hasContent) {
                    if (bubElement) {
                        bubElement.removeAttribute('data-streaming');
                        bubElement.classList.add('interrupted');
                        if (mcElement) {
                            mcElement.dataset.raw = cleanFull;
                            mcElement.innerHTML = md(cleanFull) || esc(cleanFull);
                        }
                        const note = document.createElement('div');
                        note.style.color = '#f59e0b';
                        note.style.fontSize = '12px';
                        note.style.marginTop = '8px';
                        const kindLabel = interruptKind === "TRUNCATED"
                            ? "truncated (token limit reached)"
                            : interruptKind === "STALLED"
                              ? "stalled"
                              : "interrupted";
                        note.textContent = `⚠️ Response ${kindLabel} after ${interruptTokens} tokens. Click "Continue" to resume.`;
                        bubElement.appendChild(note);
                        const acts = bubElement.querySelector('.acts') || document.createElement('div');
                        acts.className = 'acts';
                        acts.innerHTML = makeCopyRetryContinueActions(turnId);
                        bubElement.appendChild(acts);
                    }
                    turnRequests[turnId] = turnRequests[turnId] || {};
                    turnRequests[turnId].rawMarkdown = cleanFull;
                    turnRequests[turnId].interruption = { kind: interruptKind, tokens: interruptTokens };
                    const convObj = curConv();
                    if (convObj) {
                        convObj.msgs.push({ role: 'assistant', content: cleanFull });
                        save();
                    }
                    // Auto-resume on canonical server-side interruptions.
                    // Genuine client cancels (Stop / Esc / beforeunload) are
                    // handled by the earlier ctrl.signal.aborted branch, so a
                    // TRUNCATED / STALLED marker here always means the server
                    // itself hit an internal boundary the user never asked for.
                    // Cap at 50 consecutive auto-resumes so a pathological loop
                    // cannot spin forever; the user can still click Continue
                    // to resume manually after that.
                    window.__autoResume = window.__autoResume || {};
                    const _arN = (window.__autoResume[turnId] =
                                  (window.__autoResume[turnId] || 0) + 1);
                    const _willAuto = (
                        (interruptKind === 'TRUNCATED' || interruptKind === 'STALLED')
                        && !currentStreamAborted
                        && _arN <= 3
                    );
                    if (_willAuto) {
                        toast(`⚠️ Response ${interruptKind.toLowerCase()} — auto-continuing…`,
                              'warning', 2500);
                        setTimeout(() => resumeTurn(turnId), 400);
                    } else {
                        toast(`⚠️ Response ${interruptKind.toLowerCase()} — click Continue to resume`,
                              'warning', 5000);
                    }
                } else {
                    if (assistantRow) assistantRow.remove();
                    appendErrorCard(`Response ${interruptKind.toLowerCase()} before any content was generated. Please retry.`, '❌ Error', turnId);
                }
                return;
            }

            // ═══ Error / empty paths ═══
            const isErrorToken = /^\[(request timed out|error|stream cancelled|cancelled|Error:[^\]]*)\]$/i.test(full);
            if (!hasContent || !full || isErrorToken) {
                if (assistantRow) assistantRow.remove();
                appendErrorCard(
                    isErrorToken ? full.replace(/^\[|\]$/g,'') : 'Received empty response from model. Please try again.',
                    '❌ Error', turnId
                );
                return;
            }

            // ═══ Normal completion ═══
            if (assistantRow && bubElement) {
                bubElement.removeAttribute('data-streaming');
                if (mcElement) {
                        mcElement.dataset.raw = full;
                        mcElement.innerHTML = md(full) || esc(full);
                }
                const acts = bubElement.querySelector('.acts') || document.createElement('div');
                acts.className = 'acts';
                acts.innerHTML = makeCopyRetryActions(turnId);
                bubElement.appendChild(acts);
            }
            turnRequests[turnId] = turnRequests[turnId] || {};
            turnRequests[turnId].rawMarkdown = full;
            const conv = curConv();
            if (conv) {
                conv.msgs.push({ role:'assistant', content:full });
                save();
            }
        } catch (e) {
            if (ctrl.signal.aborted || currentStreamAborted) {
                return;
            } else {
                log.error('Stream error:', e);
                if (assistantRow) assistantRow.remove();
                appendErrorCard(e.message || 'Network error. Please check your connection.', '❌ Error', turnId);
            }
        } finally {
            if (timeoutId) clearTimeout(timeoutId);
            busy = false;
            ctrl = null;
            updateUIStates();
            hideThinkingIndicator();
            hideWorkIndicator();
            hideInstantTps();
        }
    };

    await executeRequest();
}

// ════════════════════════════════════════════════════════════
// send() — the primary entry point
// ════════════════════════════════════════════════════════════
async function send() {
    if (busy || !online) {
        log.warn('Cannot send: busy=' + busy + ', online=' + online);
        return;
    }
    const selVal = document.getElementById('model-sel').value;
    const readyModels = allModels.filter(m => m.status === 'ready');
    const userReadyModels = filterUserModels(readyModels);
    const hasReady = userReadyModels.length > 0;
    if (selVal !== '__smart__') {
        const selectedInfo = allModels.find(m => m.id === selVal);
        if (!selectedInfo || selectedInfo.status !== 'ready' || (isDraft(selectedInfo) && !showInternalModels)) {
            toast('⚠️ Selected model is not ready or not visible. Please choose another model.', 'warning', 4000);
            return;
        }
    } else {
        if (!hasReady) {
            toast('⚠️ No ready models. The auto‑router cannot select a model.', 'warning', 4000);
            return;
        }
    }
    const ta = document.getElementById('prompt');
    const text = ta.value.trim();
    if (!text) {
        toast('❌ Message cannot be empty', 'error', 2000);
        return;
    }
    ta.value = ''; ta.style.height = 'auto';
    const conv = ensureConv();
    conv.msgs.push({ role:'user', content:text });
    conv.ts = Date.now();
    if (conv.msgs.filter(m => m.role === 'user').length === 1) {
        conv.title = text.slice(0,50) + (text.length>50?'…':'');
        document.getElementById('tb-title').textContent = conv.title;
    }
    save();
    renderHistory(document.getElementById('hist-search').value);

    const turnId = 'turn_' + Date.now() + '_' + Math.random().toString(36).slice(2,6);
    const turnIndex = conv.msgs.length - 1;   // index of the just-pushed user message
    appendMsg('user', text, false, turnId);

    const sel = document.getElementById('model-sel').value;
    const smart = sel === '__smart__';
    let modelId = sel;
    if (smart) {
        if (userReadyModels.length > 0) {
            modelId = userReadyModels[0].id;
        } else {
            toast('No user‑ready models available', 'error');
            return;
        }
    }

    const sys = document.getElementById('sys-in').value.trim();
    // All sizing values come from the server (cached runtimeLimits) so the
    // client automatically adapts when the user swaps in a bigger GPU or a
    // smaller model. Fallbacks mirror the server defaults.
    const CTX_TOTAL    = Number(runtimeLimits.context_length) || modelContextLength || 2048;
    const SAFETY       = Number(runtimeLimits.context_safety_tokens) || 256;
    const MIN_RESPONSE = Number(runtimeLimits.min_response_tokens) || 64;
    const REC_RESPONSE = Number(runtimeLimits.recommended_response_tokens) || 1024;
    const systemTokens = sys ? estimateTokens(sys) : 0;

    // Fill history from newest to oldest, leaving room for a reply.
    const historyBudget = Math.max(128, CTX_TOTAL - MIN_RESPONSE - SAFETY);

    let used = systemTokens + 20;
    const histSlice = [];
    for (let i = conv.msgs.length - 1; i >= 0; i--) {
        const m = conv.msgs[i];
        const t = estimateTokens(m.content) + 8;
        if (used + t > historyBudget && histSlice.length >= 2) break;
        histSlice.unshift(m);
        used += t;
    }

    // 0 = unlimited. The engine slides its context window on the server
    // side; the client never asks for a bounded reply unless the caller
    // explicitly did so.
    const responseBudget = 0;

    const messages = [];
    if (sys) messages.push({ role:'system', content:sys });
    histSlice.forEach(m => messages.push({ role: m.role, content: m.content }));
    updateTokenIndicator(used, CTX_TOTAL);

    const params = {
        modelId: modelId,
        temperature: currentTemperature,
        maxTokens: responseBudget,
        turnIndex: turnIndex,
    };
    turnRequests[turnId] = {
        turnIndex: turnIndex,
        rawMarkdown: '',
        interruption: null,
        params: params,
    };

    await streamReply(turnId, messages, params, false);
}

function stopGen() {
       if (!ctrl) return;
       currentStreamAborted = true;
       ctrl.abort();
       setTimeout(() => {
         if (busy) {
           busy = false;
           updateUIStates();
           hideThinkingIndicator();
           hideInstantTps();
         }
       }, 5000);
}
function sendOrStop() {
       if (busy) {
         stopGen();
       } else {
         send();
       }
}
// ════════════════════════════════════════════════════════════ SYSTEM PROMPT TOGGLE
let systemPromptEnabled = false;
const LS_SYS_ENABLED = 'llmc_sys_enabled';
try {
       const stored = localStorage.getItem(LS_SYS_ENABLED);
       systemPromptEnabled = stored ? JSON.parse(stored) : false;
} catch(_) {
       systemPromptEnabled = false;
}
function toggleSys() {
       systemPromptEnabled = !systemPromptEnabled;
       try {
         localStorage.setItem(LS_SYS_ENABLED, JSON.stringify(systemPromptEnabled));
       } catch(_) {}
       const sysWrap = document.getElementById('sys-wrap');
       const btn = document.getElementById('btn-sys');
       if (systemPromptEnabled) {
         sysWrap.classList.add('show');
         btn.classList.add('on');
         btn.style.color = '#a78bfa';
       } else {
         sysWrap.classList.remove('show');
         btn.classList.remove('on');
         btn.style.color = '#555';
       }
}
// ════════════════════════════════════════════════════════════ UI TOGGLES
function toggleSb() {
       document.getElementById('sidebar').classList.toggle('closed');
}
// ════════════════════════════════════════════════════════════ CLEANUP ON VISIBILITY/UNLOAD
window.addEventListener('beforeunload', () => {
    busy = false;
    if (ctrl) ctrl.abort();
});
// ════════════════════════════════════════════════════════════ INIT
function init() {
       load();
       renderHistory();
       ensureConv();
       checkStatus();
       loadModels();
       fetchLimits();                          // ← initial hardware-aware limits
       setInterval(checkStatus, 10000);
       setInterval(loadModels, 60000);
       setInterval(fetchMetrics, 15000);
       setInterval(fetchLimits, 60000);        // ← refresh every 60s
       if (systemPromptEnabled) {
         const sysWrap = document.getElementById('sys-wrap');
         const btn = document.getElementById('btn-sys');
         if (sysWrap) sysWrap.classList.add('show');
         if (btn) {
           btn.classList.add('on');
           btn.style.color = '#a78bfa';
         }
       }
       const sys_text = localStorage.getItem('llmc_sys_text');
       if (sys_text) document.getElementById('sys-in').value = sys_text;
       document.getElementById('sys-in').addEventListener('change', e => {
         localStorage.setItem('llmc_sys_text', e.target.value);
       });
       const btnThinking = document.getElementById('btn-thinking');
       if (btnThinking) {
         btnThinking.textContent = thinkingModeEnabled ? '💭 Thinking ON' : '💭 Thinking OFF';
         btnThinking.classList.toggle('on', thinkingModeEnabled);
       }
       const internalBtn = document.getElementById('btn-internal');
       if (internalBtn) {
         internalBtn.textContent = showInternalModels ? '🔍 Internal ON' : '🔍 Internal OFF';
         internalBtn.classList.toggle('on', showInternalModels);
       }
       ModalManager.init();
       log.info('UI initialized');
}
if (document.readyState === 'loading') {
       document.addEventListener('DOMContentLoaded', init);
} else {
       init();
}
</script>
</body>
</html>
"""
