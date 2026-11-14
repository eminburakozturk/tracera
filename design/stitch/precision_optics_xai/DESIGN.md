---
name: Precision Optics XAI
colors:
  surface: '#0c1322'
  surface-dim: '#0c1322'
  surface-bright: '#323949'
  surface-container-lowest: '#070e1d'
  surface-container-low: '#141b2b'
  surface-container: '#191f2f'
  surface-container-high: '#232a3a'
  surface-container-highest: '#2e3545'
  on-surface: '#dce2f7'
  on-surface-variant: '#bcc9cd'
  inverse-surface: '#dce2f7'
  inverse-on-surface: '#293040'
  outline: '#869397'
  outline-variant: '#3d494c'
  surface-tint: '#4cd7f6'
  primary: '#4cd7f6'
  on-primary: '#003640'
  primary-container: '#06b6d4'
  on-primary-container: '#00424f'
  inverse-primary: '#00687a'
  secondary: '#5de6ff'
  on-secondary: '#00363e'
  secondary-container: '#00cbe6'
  on-secondary-container: '#00515d'
  tertiary: '#6cd3f7'
  on-tertiary: '#003543'
  tertiary-container: '#46b3d5'
  on-tertiary-container: '#004253'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#acedff'
  primary-fixed-dim: '#4cd7f6'
  on-primary-fixed: '#001f26'
  on-primary-fixed-variant: '#004e5c'
  secondary-fixed: '#a2eeff'
  secondary-fixed-dim: '#2fd9f4'
  on-secondary-fixed: '#001f25'
  on-secondary-fixed-variant: '#004e5a'
  tertiary-fixed: '#b7eaff'
  tertiary-fixed-dim: '#6cd3f7'
  on-tertiary-fixed: '#001f28'
  on-tertiary-fixed-variant: '#004e61'
  background: '#0c1322'
  on-background: '#dce2f7'
  surface-variant: '#2e3545'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 3rem
    fontWeight: '600'
    lineHeight: 3.5rem
    letterSpacing: -0.025em
  display-lg-mobile:
    fontFamily: Inter
    fontSize: 2rem
    fontWeight: '600'
    lineHeight: 2.5rem
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 2rem
    fontWeight: '600'
    lineHeight: 2.5rem
    letterSpacing: -0.02em
  headline-lg-mobile:
    fontFamily: Inter
    fontSize: 1.5rem
    fontWeight: '600'
    lineHeight: 2rem
    letterSpacing: -0.015em
  headline-sm:
    fontFamily: Inter
    fontSize: 1.25rem
    fontWeight: '500'
    lineHeight: 1.75rem
    letterSpacing: -0.01em
  body-lg:
    fontFamily: Inter
    fontSize: 1rem
    fontWeight: '400'
    lineHeight: 1.5rem
  body-md:
    fontFamily: Inter
    fontSize: 0.875rem
    fontWeight: '400'
    lineHeight: 1.25rem
  body-sm:
    fontFamily: Inter
    fontSize: 0.75rem
    fontWeight: '400'
    lineHeight: 1rem
  code-md:
    fontFamily: JetBrains Mono
    fontSize: 0.875rem
    fontWeight: '500'
    lineHeight: 1.25rem
  label-data:
    fontFamily: JetBrains Mono
    fontSize: 0.75rem
    fontWeight: '500'
    lineHeight: 1rem
    letterSpacing: 0.05em
  label-xs:
    fontFamily: Inter
    fontSize: 0.6875rem
    fontWeight: '600'
    lineHeight: 0.875rem
    letterSpacing: 0.04em
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  gutter: 1rem
  gutter-lg: 1.5rem
  margin: 1rem
  margin-md: 1.5rem
  margin-lg: 2rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 1rem
  space-lg: 1.5rem
  space-xl: 2rem
---

## Brand & Style
The design system embodies the clinical exactitude of high-throughput electronics manufacturing alongside the distilled calm of premium consumer hardware. Built for hardware reliability engineers, computer vision specialists, and factory floor technicians, the interface balances ultra-dense diagnostic telemetry with Apple-inspired clarity.

Key style pillars:
- **Clinical Rigor:** Every visual element serves diagnostic legibility. Zero decorative clutter or unanchored gradients; UI elements emulate physical optical metrology displays.
- **Explainable Precision:** Deep inspection data, confidence vectors, and saliency heatmaps are integrated with hairline precision, using tabular alignment and optical hierarchy to minimize cognitive fatigue during rapid verdict cycles.
- **Bilingual Typographic Nuance:** Full structural support for both international microelectronics nomenclature and Turkish engineering vocabulary (proper dotting/casing for `İ`/`i` and `I`/`ı`, uppercase diacritics, and strict ISO/IEC unit formatting).

## Colors
The color architecture supports rapid scanning under varying factory lighting conditions, centered on a low-luminance, high-contrast dark environment by default, with an equivalent clinical white room mode.

### Base Canvas & Neutral Surfaces
- **Dark Mode (Default):** Canvas `#0b0f17`, Surface `#111827`, Raised Surface `#1e293b`. Muted borders use 1px hairlines at `rgba(255, 255, 255, 0.08)`.
- **Light Mode (Clinical White Room):** Canvas `#f8fafc`, Surface `#ffffff`, High-Contrast Text/Base `#0f172a`. Hairline borders use `rgba(15, 23, 42, 0.08)`.

### Primary & Accent
- **Electric Cyan (`#06b6d4` / `#22d3ee` / `#0891b2`):** Reserved for focal UI elements, active camera feeds, optical bounding boxes, laser telemetry readouts, and affirmative navigation states.

### Quality Verdict Spectrum
Verdict colors are strictly functional indicators and must never be repurposed for generic styling:
- **Pass / Nominal:** `#10b981` (Emerald Green)
- **Warning / Review Required:** `#f59e0b` (Industrial Amber)
- **Fail / Critical Defect:** `#ef4444` (Vivid Red)

### Defect Taxonomy Coding
Six calibrated chromatic identifiers correspond to neural segmentation layers:
- `missing_hole`: Sky Blue (`#38bdf8`)
- `mouse_bite`: Soft Coral Pink (`#f472b6`)
- `open_circuit`: Laser Yellow (`#facc15`)
- `short`: Terminal Red-Orange (`#f87171`)
- `spur`: Violet Precision (`#a78bfa`)
- `spurious_copper`: Mint Jade (`#34d399`)

## Typography
Typographic execution prioritizes uncompromised data legibility. Inter delivers an authoritative, neutral human-interface standard, while JetBrains Mono provides exact columnation for industrial metrics, coordinates (X/Y mils), inference confidence scores, and firmware logs.

Implementation requirements:
- **Tabular Figures:** All numeric metrics, percentages, tolerances, and defect tallies must activate OpenType tabular numbers (`tnum`) and slashed zero (`zero`) to eliminate layout jitter during high-rate inspection telemetry.
- **Turkish Engineering Locale:** Rendering engines must explicitly maintain correct Turkish character sets (`lang="tr"`), ensuring that capitalization transforms (`i` to `İ`, `ı` to `I`) preserve morphological clarity across defect terms like *İletken Hat Kopukluğu* (Open Circuit) or *Kısa Devre* (Short Circuit).

## Layout & Spacing
The layout follows a rigorous 8pt spatial grid (with a 4pt sub-grid for tight metric clusters and badge elements).

- **Grid Architecture:** 12-column dynamic fluid grid on workstation viewports (>= 1280px), collapsing to 8 columns on industrial inspection tablets (768px - 1279px), and 4 columns on hand-held diagnostic readers (< 768px).
- **Split-Pane Metrology Viewports:** Machine vision inspection panels reserve a non-collapsing 60% viewport width for the primary high-resolution board canvas, while diagnostic telemetry, defect breakdown, and XAI saliency heatmaps occupy a standardized 40% dockable sidebar.
- **Internal Component Spacing:** Strict micro-spacing ensures that sensor tables and optical readouts pack maximum actionable intelligence into dense dashboards without vertical or horizontal bleed.

## Elevation & Depth
Depth is constructed through clinical tonal tiering and razor-sharp border separation rather than heavy skeuomorphic shadows, emulating the look of premium high-resolution optical workstations.

- **Tonal Layers:** The lowest stratum is the workspace ground (`#0b0f17`), followed by panel surfaces (`#111827`), elevated diagnostic flyouts (`#1e293b`), and hovering inspector reticles (`#334155`).
- **Hairline Borders:** Every card, viewport, floating palette, and inspector panel is bound by a crisp 1px outline: `rgba(255, 255, 255, 0.08)` in dark mode and `rgba(15, 23, 42, 0.08)` in light mode. This eliminates blurred edges when rendered on 4K/5K industrial retina displays.
- **Backdrop Diffusion:** Modal dialogs, HUD overlays, and floating optical controls employ a 16px to 24px backdrop blur (`backdrop-filter: blur(20px)`) combined with a 75% alpha fill of `#111827`, establishing distinct focus above high-density circuit scans.
- **Micro-Glow Feedback:** When selecting a specific PCB trace or neural heatmap segment, standard blur drop-shadows are replaced with a targeted 0 0 12px neon halo tinted precisely to the corresponding defect class or verdict cyan.

## Shapes
Geometry strikes an intentional equilibrium between ergonomic consumer elegance and technical precision.
- **Primary Cards & Telemetry Containers:** Standardized at 1rem (`rounded-lg` / 16px corner radius), echoing precision aluminum hardware enclosures.
- **Buttons, Micro-Badges & Filter Chips:** Employ 0.5rem (`rounded` / 8px corner radius) for crisp, tool-like tactility.
- **Target Bounding Reticles:** Inner optical focus frames and PCB bounding boxes maintain completely square corners (`0px` radius) with accented corner brackets to preserve geometric calibration.

## Components

### Buttons & Quick Actions
- **Primary Diagnostic Action:** Electric Cyan background (`#06b6d4`), solid slate text (`#0b0f17`), font weight 600, 8px radius. Active state scales subtly (`scale: 0.98`) with no bouncy transitions.
- **Secondary Tooling Button:** Subtle surface fill (`#1e293b`), hairline border (`rgba(255, 255, 255, 0.08)`), white text (`#f8fafc`). Hover introduces a 1px border highlight matching cyan (`#22d3ee`).
- **Critical Override (Reject Lot):** Solid verdict red fill (`#ef4444`) with high-contrast white text, reserved exclusively for manual fail confirmations.

### Inspection Verdict Cards (16px Radius)
- Built on `#111827` backing, framed by a continuous 1px hairline border. The card header features a 2px colored status indicator matching the verdict (Pass: Green, Warning: Amber, Fail: Red).
- Card interiors feature monospace tabular readouts displaying batch number, SMT line identifier, drift delta, and aggregate inspection confidence (e.g., `99.84% CONF`).

### Defect Class Chips & Toggles
- Compact, dense pills (8px radius) featuring a persistent 6px circular dot tinted with the defect class color (e.g., `#f87171` for `short`).
- Unselected state renders with low opacity (`opacity: 0.45`). Active state introduces a matching translucent fill (`background: rgba(defect_color, 0.12)`) and fully opaque text.

### Telemetry Tables & Anomaly Lists
- Rows have a fixed height of 36px for dense inspection or 48px for standard review.
- Columns align strictly: text left, numbers and timestamps right-aligned in tabular `JetBrains Mono`. Hovering a row highlights the trace boundary on the synchronized PCB viewport.

### Checkboxes, Radios, and Switches
- High-precision mechanical look: 16px checkboxes with a 4px corner radius. Selected state transitions to solid `#06b6d4` with a crisp white check vector.
- Toggle switches: 36px x 20px pill container with a 16px circular actuator, sliding across a smooth 150ms linear transition.

### Optical & XAI Viewport Overlay
- HUD elements floating over the raw PCB camera feed feature a frosted dark backplate (`rgba(17, 24, 39, 0.85)` with `backdrop-filter: blur(12px)`).
- Heatmap opacity sliders, zoom calibration markers (`1.0x`, `5.0x`, `20.0x`), and CAM (Class Activation Mapping) toggle groups are docked to bottom corners with 16px internal padding and hairline dividers.