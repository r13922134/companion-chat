import React from "react";
import { createRoot } from "react-dom/client";
import { flushSync } from "react-dom";
import LiquidGlass from "liquid-glass-react";

const LIQUID_GLASS_EXAMPLE_PROPS = {
  displacementScale: 220,
  blurAmount: 0.22,
  saturation: 185,
  aberrationIntensity: 4,
  elasticity: 0.42,
  cornerRadius: 100,
  overLight: true,
  mode: "shader",
};

const GLASS_KIND_LAYOUT = {
  input: {
    padding: "0",
    props: {
      displacementScale: 340,
      blurAmount: 0.38,
      saturation: 220,
      aberrationIntensity: 7,
      elasticity: 0.24,
    },
  },
  cluster: { padding: "16px" },
  status: { padding: "0" },
};

function Glass({ kind, className = "", interactive = false, children }) {
  const classSuffix = `liquid-control-glass--${kind}${className ? ` ${className}` : ""}`;
  const isActionPill = className.includes("liquid-control-glass--actions");
  const layout = GLASS_KIND_LAYOUT[kind] || GLASS_KIND_LAYOUT.cluster;
  const padding = isActionPill ? "13px 20px" : layout.padding;
  const handleGlassClick = interactive ? () => {} : undefined;

  return (
    <div className={`liquid-glass-shell ${classSuffix}`}>
      <LiquidGlass
        {...LIQUID_GLASS_EXAMPLE_PROPS}
        {...(layout.props || {})}
        className={`liquid-control-glass ${classSuffix}`}
        onClick={handleGlassClick}
        padding={padding}
        style={{ position: "absolute", top: "50%", left: "50%" }}
      >
        {children}
      </LiquidGlass>
    </div>
  );
}

function SendIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M5 12h14" />
      <path d="M13 6l6 6-6 6" />
    </svg>
  );
}

function BackIcon() {
  return (
    <svg fill="none" viewBox="0 0 24 24" strokeWidth="2.2" stroke="currentColor" aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 19.5 8.25 12l7.5-7.5" />
    </svg>
  );
}

function ResultIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <path d="M4 19V9M10 19V5M16 19v-7M22 19H2" />
    </svg>
  );
}

function MicIcon() {
  return (
    <svg className="voice-orb-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 18.75a6 6 0 0 0 6-6v-1.5m-6 7.5a6 6 0 0 1-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 0 1-3-3V4.5a3 3 0 1 1 6 0v8.25a3 3 0 0 1-3 3Z" />
      <path className="mic-slash" strokeLinecap="round" strokeLinejoin="round" d="M4.8 4.8 19.2 19.2" />
    </svg>
  );
}

function EndContent() {
  return (
    <>
      <span className="pill-dots" aria-hidden="true"><i></i><i></i><i></i></span>
      <span>結束</span>
    </>
  );
}

function TextInputControl() {
  return (
    <Glass kind="input" interactive>
      <form className="text-input-form" id="textInputForm" autoComplete="off">
        <input className="text-input" id="textInput" type="text" placeholder="輸入文字" autoComplete="off" disabled aria-label="輸入文字訊息" />
        <button className="call-btn text-send-btn" id="sendTextButton" type="submit" disabled title="送出文字" aria-label="送出文字">
          <SendIcon />
        </button>
      </form>
    </Glass>
  );
}

function GlassButton({ className = "", children }) {
  return (
    <Glass kind="cluster" className={`liquid-control-glass--button ${className}`} interactive>
      <div className="control-cluster">
        {children}
      </div>
    </Glass>
  );
}

function MainLeftControls() {
  return (
    <GlassButton className="liquid-control-glass--compact">
      <span className="call-btn back-btn" id="chooseAgainButton" role="button" tabIndex="-1" data-disabled aria-disabled="true" title="返回" aria-label="返回">
        <BackIcon />
      </span>
    </GlassButton>
  );
}

function MicControl() {
  return (
    <span className="call-btn mic-btn" id="muteMicButton" role="button" tabIndex="-1" data-disabled aria-disabled="true" title="靜音" aria-label="靜音" aria-pressed="false">
      <MicIcon />
    </span>
  );
}

function EndControl() {
  return (
    <span className="call-btn end-btn" id="endSessionButton" role="button" tabIndex="0" aria-disabled="false" title="結束對話" aria-label="結束對話">
      <EndContent />
    </span>
  );
}

function MainActionControls() {
  return (
    <div className="call-actions-group">
      <GlassButton className="liquid-control-glass--mic">
        <MicControl />
      </GlassButton>
      <GlassButton className="liquid-control-glass--compact liquid-control-glass--dark">
        <span className="call-btn result-btn hidden" id="depressionResultButton" role="button" tabIndex="0" aria-disabled="false" title="查看憂鬱評估結果" aria-label="查看憂鬱評估結果">
          <ResultIcon />
        </span>
      </GlassButton>
      <GlassButton className="liquid-control-glass--actions">
        <EndControl />
      </GlassButton>
    </div>
  );
}

function FeedbackActionControls() {
  return (
    <div className="call-actions-group">
      <GlassButton className="liquid-control-glass--mic">
        <MicControl />
      </GlassButton>
      <GlassButton className="liquid-control-glass--actions liquid-control-glass--end-only">
        <EndControl />
      </GlassButton>
    </div>
  );
}

function RealtimeControls({ variant }) {
  const isMain = variant === "main";
  return (
    <div className="call-bottom-bar" data-liquid-glass-react="mounted">
      <TextInputControl />
      <div className="call-controls-row">
        {isMain ? <MainLeftControls /> : null}
        {isMain ? <MainActionControls /> : <FeedbackActionControls />}
      </div>
    </div>
  );
}

function StatusPillGlass({ id, className, initialText = "" }) {
  return (
    <Glass kind="status" className={`liquid-control-glass--${id}`}>
      <span className={className} id={id}>{initialText}</span>
    </Glass>
  );
}

function CallStatusGlass() {
  return (
    <Glass kind="status" className="liquid-control-glass--call-status">
      <div className="call-status-label" id="callStatusLabel" style={{ opacity: 0, visibility: "hidden" }}></div>
    </Glass>
  );
}

const mountedRoots = new WeakMap();

function mount(target, options = {}) {
  if (!target) return null;
  if (mountedRoots.has(target)) return mountedRoots.get(target);
  const variant = options.variant || target.dataset.variant || "main";
  const root = createRoot(target);
  flushSync(() => {
    root.render(<RealtimeControls variant={variant} />);
  });
  target.dataset.reactMounted = "true";
  mountedRoots.set(target, root);
  return root;
}

function mountStatusPill(target, options = {}) {
  if (!target) return null;
  if (mountedRoots.has(target)) return mountedRoots.get(target);
  const initialText = options.initialText || target.dataset.initialText || target.textContent || "未連線";
  const root = createRoot(target);
  flushSync(() => {
    root.render(<StatusPillGlass id="statusPill" className="status-pill" initialText={initialText.trim()} />);
  });
  target.dataset.reactMounted = "true";
  mountedRoots.set(target, root);
  return root;
}

function mountCallStatus(target) {
  if (!target) return null;
  if (mountedRoots.has(target)) return mountedRoots.get(target);
  const root = createRoot(target);
  flushSync(() => {
    root.render(<CallStatusGlass />);
  });
  target.dataset.reactMounted = "true";
  mountedRoots.set(target, root);
  return root;
}

window.CompanionRealtimeControls = {
  mount,
  mountStatusPill,
  mountCallStatus,
};
