# CCHP Paper Figure Prompt Pack

Timestamp: `2026-04-01 Asia/Shanghai`

## Purpose

This note packages figure ideas for the current CCHP manuscript and turns them into direct prompts for Nano Banana style image generation.
The target is not decorative art.
The target is presentation-ready scientific figure ideation: vivid, professional, structurally clear, easier to understand than a pure journal schematic, and still faithful to the actual plant, control logic, and cost mechanism described in the paper.

This pack is aligned with:

- `cchp-paper/sections/01-introduction.tex`
- `cchp-paper/sections/02-system-description.tex`
- `cchp-paper/sections/03-mathematical-model.tex`
- the visual role played by Fig. 1 and Fig. 2 in `reference/pdf/Deep reinforcement learning for joint dispatch of battery storage and gas.pdf`

## Chosen Color Scheme

Palette `A: Okabe-Ito academic standard`

- Main structural blue: `#0072B2`
- Heat-path orange: `#E69F00`
- Cooling-path green: `#009E73`
- Alert / penalty accent: `#D55E00`
- Background grey: `#F7F7F7`
- Standard outline grey: `#CCCCCC`
- Text charcoal: `#333333`
- Connector dark grey: `#4D4D4D`

Recommended semantic color mapping for all figures:

- Electricity / electrical path: blue
- Heating / recovered heat / TES / boiler path: orange
- Cooling / ABS-ECH cooling path: green
- Violations / unmet load / impossible action / penalty: vermillion

## Global Style Rules For All Prompts

Use these rules for every figure, even if the prompt below already repeats some of them.

1. The figure should look like a polished scientific presentation slide illustration, not a raw journal line drawing, not a marketing poster, and not a photorealistic rendering.
2. Keep the composition clean and structured like a good research-group PPT: strong title hierarchy, clear grouped regions, visual focal points, and concise labels.
3. For every physical device, prefer a recognizable facility sketch icon instead of a plain abstract rectangle.
4. Use soft white or very light grey backgrounds, thin-to-medium connector lines, and restrained color accents from the chosen palette.
5. Mild 2.5D depth, soft shadows, or gentle gradients are allowed only to make the equipment icons more readable and presentation-like.
6. Avoid glossy commercial infographic style, lens flares, cartoon mascots, overloaded icon packs, and dramatic cinematic lighting.
7. Keep all component names technically grounded and consistent with the manuscript.
8. Prefer a horizontal left-to-right scientific layout unless explicitly stated otherwise.
9. Make labels readable and concise enough that a human can later redraw the figure in PowerPoint, Illustrator, Inkscape, or Visio.
10. Treat the AI result as a composition sketch for later manual cleanup, not as the final submission-ready artwork.

## Physical Facility Sketch Icon Rule

Whenever a figure contains plant equipment, the prompt should explicitly ask the model to draw the component as a recognizable physical facility sketch icon rather than a plain box.

Use these icon intentions consistently:

- Grid: transmission tower, substation cabinet, or grid interface icon
- Renewables: solar panel field and/or wind turbine sketch
- Gas turbine: compact industrial gas turbine package or turbine-generator skid
- HRSG: boxy heat-recovery unit with heat-exchanger fins or ducted recovery boiler appearance
- Boiler: industrial boiler vessel or burner-equipped thermal unit sketch
- BES: battery cabinet / battery rack / containerized battery energy storage icon
- TES: insulated cylindrical thermal storage tank or hot-water storage vessel
- ABS: absorption chiller unit with heat-driven cooling identity
- ECH: electric chiller or compressor chiller unit
- Electric bus / thermal node / cooling node: can remain abstract hubs if needed
- Loads: simple building, campus block, or demand-side facility icon when visual clarity benefits

The icon style should stay unified:

- clean engineering sketch
- presentation-friendly PPT look
- not photorealistic
- not hand-drawn messy doodle
- not childish cartoon
- simplified physical silhouette with technical credibility

## Figure Set Overview

This pack proposes four figures.

1. Intro overview figure
2. Simplified plant layout for Section 2
3. Physics-aware DRL and safety-projection pipeline for Section 3
4. Dynamic GT O&M allocation mechanism for Section 3

---

## Figure 1

### Working Title

`Problem framing and paper overview for reliable economic dispatch in a coupled CCHP system`

### Intended Location

Introduction, preferably after the paragraph that introduces the multi-energy coupling challenge and before the contribution summary.

### Main Message

This figure should give readers a fast first impression of the whole paper:

- what the physical plant looks like
- why electricity, heat, and cooling decisions are coupled
- why naive DRL is risky
- what the proposed physics-in-the-loop DRL framework actually does
- what outcomes the paper evaluates

This is the best candidate for an introduction figure.
It should not be too detailed.
It should behave like a conceptual map of the paper.

### Mandatory Visual Elements

- A compact simplified plant sketch in the center:
  - grid
  - renewables
  - gas turbine
  - HRSG
  - boiler
  - BES
  - TES
  - absorption chiller
  - electric chiller
  - electric load
  - heating load
  - cooling load
- Three visibly different energy pathways:
  - electricity path
  - heat path
  - cooling path
- A control layer above the plant:
  - observation
  - DRL policy
  - physics-aware correction / safety projection
  - executable dispatch
- A small problem block on the left:
  - coupled multi-energy dispatch
  - physical feasibility
  - annual cost minimization
  - reliability constraints
- A small contribution / output block on the right:
  - reliable same-information DRL comparison
  - yearly evaluation
  - multi-seed robustness
  - planner-side Oracle as reference

### Recommended Caption

`Overview of the studied CCHP dispatch problem and the role of the proposed physics-in-the-loop DRL framework. The controller observes the coupled plant state, issues normalized actions, and executes only the physics-feasible dispatch after a safety-aware projection step.`

### Direct English Prompt

```text
Create a presentation-ready scientific overview figure for a research-group PPT on physics-informed deep reinforcement learning for reliable economic dispatch of a coupled cooling, heating, and power system. The style should be cleaner and more visual than a journal line diagram, but still technically honest and academically professional. Use Okabe-Ito academic colors only: structural blue #0072B2, heat-path orange #E69F00, cooling-path green #009E73, alert accent #D55E00, light grey background panels #F7F7F7, white or near-white modules, charcoal text #333333, connector lines in dark grey #4D4D4D. Allow light presentation-style depth, soft shadow, and gentle gradient only where needed to make icons legible.

The composition should be a wide horizontal overview with three grouped regions.

LEFT REGION, title in small caps: "Dispatch challenge". Use a faint grey panel. Inside it place four concise problem blocks with small technical icons and short labels: "Coupled electricity-heat-cooling decisions", "Physical feasibility of thermally driven cooling", "Long-horizon annual operating cost", "Reliability constraints for electric, heat, and cooling supply". Keep these as compact scientific callout cards in PPT style, not childish icons.

CENTER REGION, title in small caps: "Studied CCHP plant". Show a simplified but technically grounded plant layout using recognizable physical facility sketch icons instead of generic boxes. Grid should look like a transmission or substation icon, renewables should look like solar panels or wind turbines, GT should look like a compact industrial turbine package, HRSG should look like a heat-recovery unit, Boiler should look like an industrial boiler, BES should look like battery cabinets or a containerized battery system, TES should look like an insulated storage tank, ABS should look like a heat-driven chiller unit, and ECH should look like an electric chiller unit. Electric load, heating load, and cooling load may be shown as simple campus or building demand icons. Draw electricity flow in blue, heat flow in orange, and cooling flow in green. The GT should send electricity toward the electrical bus and exhaust heat toward the HRSG. The HRSG, Boiler, and TES should feed the heating branch. The ABS should receive driving heat and produce cooling. The ECH should receive electricity and produce cooling. Cooling demand should be satisfied by both ABS and ECH. BES should connect to the electrical bus. TES should connect to the thermal branch. Grid and renewables should connect to the electrical bus. Arrange the plant to clearly show three coupled pathways rather than a messy network.

TOP OVERLAY above or below the center plant, title in small caps: "Control framework". Show a sequence of clean rounded PPT modules: "State observation", "DRL policy", "Physics-aware safety projection", "Executable dispatch". Use blue borders for generic control modules and add a special mixed blue-orange-green accent for the safety projection box. Add thin arrows left to right. Under the safety projection box add a small note: "raw action -> feasible action". Add a feedback arrow from plant outcomes back to state observation.

RIGHT REGION, title in small caps: "Paper outputs". Use a faint grey panel with four result boxes: "Yearly cost and reliability evaluation", "Same-information DRL comparison", "Multi-seed robustness analysis", "Oracle benchmark as planner-side reference". Use precise engineering text, no hype language.

At the bottom center include a very compact legend with three line swatches labeled "electricity", "heat", and "cooling". Add a small note below the figure: "Executable absorption cooling depends on available driving heat and thermal state."

Style requirements: high-quality scientific presentation illustration, visually clearer than a pure paper schematic, crisp vector-like edges, recognizable physical facility sketch icons, dense but readable labeling, mild PPT-style depth allowed, no photorealism, no glossy business infographic look, no exaggerated 3D perspective, no cartoon style, no dark background, no purple theme. Make the layout orderly, factual, professional, and easy to redraw manually for a final paper figure.
```

---

## Figure 2

### Working Title

`Simplified plant layout and coupled energy pathways of the studied CCHP system`

### Intended Location

Section 2, close to `Physical configuration` and `Energy pathways and operational roles`.

### Main Message

This figure plays the same role as the reference paper's `Simplified plant layout`, but it must be more clearly multi-carrier.
It should teach readers how the plant physically works before any equations appear.

### Mandatory Visual Elements

- A central electric bus
- GT connected to the electric bus
- GT exhaust heat to HRSG
- HRSG to thermal bus
- boiler to thermal bus
- TES connected bidirectionally with the thermal bus
- ABS receives heat from thermal bus and outputs cooling
- ECH receives electricity from electric bus and outputs cooling
- cooling demand node fed by ABS and ECH
- electric demand node fed by GT, BES, renewables, and grid
- heating demand node fed by HRSG, boiler, and TES discharge
- BES charge/discharge on the electric side
- optional small callout: ABS cooling is available only when sufficient driving heat exists

### Important Design Advice

- Avoid trying to draw a thermodynamic process diagram with pipes, valves, and turbines in mechanical detail.
- Use a simplified engineering system diagram with recognizable physical equipment sketch icons.
- The key is energy pathways and coupling, but each device should still look like its real-world facility category.

### Recommended Caption

`Simplified layout of the studied grid-connected CCHP plant. Electricity, heating, and cooling are coupled through the GT-HRSG chain, BES, TES, boiler support, and the dual cooling paths provided by the absorption and electric chillers.`

### Direct English Prompt

```text
Create a publication-quality scientific presentation diagram for a research PPT that explains the physical configuration of a grid-connected coupled cooling, heating, and power system. The figure should look more visual and more intuitive than a plain journal line drawing, while still staying technically accurate. Use the Okabe-Ito academic palette. Electricity path must use blue #0072B2, heat path must use orange #E69F00, cooling path must use green #009E73, text charcoal #333333, outlines light grey #CCCCCC, connector lines dark grey #4D4D4D, background mostly white with extremely faint grey grouping panels. Allow mild 2.5D presentation depth and soft shadows for the equipment icons only.

Arrange the figure as a wide left-to-right system layout with three horizontal layers or aligned pathway zones.

MANDATORY COMPONENTS:
- Grid
- Renewables
- Electric bus
- Gas Turbine (GT)
- Battery Energy Storage (BES)
- Heat Recovery Steam Generator (HRSG)
- Boiler
- Thermal Energy Storage (TES)
- Absorption Chiller (ABS)
- Electric Chiller (ECH)
- Electric load
- Heating load
- Cooling load

For every physical component, use a corresponding facility sketch icon rather than a generic box. Grid should look like a transmission tower or substation. Renewables should look like a solar panel array and/or wind turbine. GT should look like a gas-turbine package. BES should look like battery racks or a battery container. HRSG should look like a heat-recovery unit. Boiler should look like an industrial boiler. TES should look like a cylindrical insulated tank. ABS should look like a heat-driven absorption chiller. ECH should look like an electric chiller. Loads can look like simple building or campus demand icons.

MANDATORY CONNECTION LOGIC:
- Grid and Renewables feed the electric bus.
- GT feeds the electric bus with electrical power.
- BES connects bidirectionally to the electric bus with charge and discharge arrows.
- GT exhaust heat flows to HRSG.
- HRSG output, Boiler output, and TES discharge feed a thermal bus or thermal allocation node.
- TES also has a charging arrow from the thermal bus back into storage.
- ABS receives driving heat from the thermal bus and outputs cooling to the cooling load node.
- ECH receives electricity from the electric bus and outputs cooling to the cooling load node.
- Electric load is supplied from the electric bus.
- Heating load is supplied from the thermal bus.
- Cooling load is supplied jointly by ABS cooling and ECH cooling.

VISUAL EMPHASIS:
- Make the electric, heat, and cooling pathways visually distinct with the three semantic colors.
- Show that ABS competes for heat with direct heating and TES charging.
- Add a small callout box near ABS: "thermally driven cooling, feasible only with sufficient driving heat".
- Add a small callout box near ECH: "electrically driven cooling fallback".
- Add a small callout box near TES: "short-term heat buffering and shifting".
- Add a small callout box near BES: "electrical flexibility and grid-shaping support".

LABEL STYLE:
- Use concise scientific labels, not full sentences inside every box.
- Use small-caps section tags above grouped regions such as "Electrical pathway", "Thermal pathway", "Cooling pathway".
- Include a tiny legend showing blue = electricity, orange = heat, green = cooling.

STYLE REQUIREMENTS:
scientific PPT-style engineering illustration, facility sketch icons, no photorealistic machines, no glossy commercial infographic look, no decorative background, no fake CAD rendering, no excessive color filling, no cartoon look, no dark mode. The figure should look like something suitable for a serious lab presentation and still be truthful, structured, and easy to manually redraw for final publication.
```

---

## Figure 3

### Working Title

`Physics-aware DRL dispatch pipeline with safety projection and executable action generation`

### Intended Location

Section 3, near the MDP formulation and the definition of the safety projection operator.

### Main Message

This figure explains the algorithmic novelty in an intuitive way.
It should show that the policy emits a raw action, but the environment executes a corrected feasible action after physics-aware projection.
This is the figure that most directly explains the difference between plain DRL and your framework.

### Mandatory Visual Elements

- Left input block:
  - exogenous variables
  - plant state
  - time encoding
- observation vector enters DRL policy
- raw six-dimensional action vector:
  - GT request
  - BES request
  - boiler request
  - ABS request
  - ECH request
  - TES request
- physics-aware safety projection block
- inside the projection block, visibly show three ordered sub-stages:
  - actuator clipping and GT ramp envelope
  - heat-first allocation and thermal feasibility check
  - residual cooling completion by ECH
- executable action goes into plant / environment
- environment transition returns:
  - next state
  - operating cost
  - unmet heat / unmet cooling
  - violation indicators
  - reward

### Recommended Caption

`Schematic of the physics-aware DRL dispatch pipeline. The policy outputs normalized raw control requests, which are converted into executable plant actions only after a safety-aware projection step that enforces actuator bounds, thermal feasibility, and residual cooling completion.`

### Direct English Prompt

```text
Create a detailed scientific presentation pipeline figure for a research PPT on physics-informed deep reinforcement learning for CCHP dispatch. The figure should explain the control logic clearly enough for an energy systems reader who is not a reinforcement learning specialist. Use a clean horizontal flowchart layout, white rounded modules, precise arrows, concise labels, and the Okabe-Ito academic palette: blue #0072B2, orange #E69F00, green #009E73, alert vermillion #D55E00, charcoal text #333333, faint grey panel #F7F7F7. Keep it visually richer than a raw line flowchart but still technically disciplined.

LAYOUT:
Use five major stages from left to right:
1. Observation construction
2. DRL policy
3. Physics-aware safety projection
4. Executable plant dispatch
5. Environment feedback and reward

STAGE 1, "Observation construction":
Create a grouped input panel with three stacked white boxes:
- "Exogenous variables" containing small labels like load, renewables, prices, weather
- "Plant state" containing small labels like BES state, GT state, TES state, temperature and feasibility margins
- "Time encoding" containing small labels like hour-of-day and periodic features
These should merge into a single blue-bordered box labeled "State observation vector s_t".

STAGE 2, "DRL policy":
Show a compact blue module labeled "Policy network". Under it show a small vector label for the raw six-dimensional action:
"a_t = [u_gt, u_bes, u_boiler, u_abs, u_ech, u_tes]".
Add a tiny note: "normalized control requests".

STAGE 3, "Physics-aware safety projection":
This is the key visual center. Use a slightly larger box with a highlighted border combining blue, orange, and green accents. Inside it place three ordered sub-blocks connected vertically or left-to-right:
- "Actuator clipping and GT ramp envelope"
- "Heat-first allocation and thermal feasibility check"
- "Residual cooling completion by ECH"
At the output of this block show the transformed action label "a_t -> a_t_tilde".
Add a compact note in small type: "raw action is not executed directly".

STAGE 4, "Executable plant dispatch":
Show a simplified plant execution block with miniature physical facility sketch icons for GT, BES, HRSG, Boiler, TES, ABS, and ECH, not generic small boxes. GT should resemble a turbine package, BES a battery cabinet, HRSG a recovery unit, Boiler an industrial boiler, TES a storage tank, ABS a heat-driven chiller, and ECH an electric chiller. The output should be labeled "executable dispatch". Include tiny colored arrows inside this block to indicate electricity, heat, and cooling pathways.

STAGE 5, "Environment feedback and reward":
Show a grouped output panel containing:
- next state s_(t+1)
- operating cost C_t
- unmet heat and unmet cooling
- violation indicators
- reward r_t = -C_t
Include a feedback arrow returning from this panel back to the observation stage.

ADDITIONAL CALLOUTS:
- Near the safety projection stage add a vermillion-bordered small box labeled "prevents physically impossible cooling requests".
- Near the reward panel add a small note: "reliability-sensitive yearly dispatch objective".

STYLE:
serious scientific PPT flowchart, crisp vector-like lines, small facility sketch icons inside the plant-dispatch stage, no photorealism, no neural network eye-candy, no exaggerated deep-learning aesthetic, no random 3D cubes, no circuit-board style. The figure must feel factual, method-oriented, and honest to the described CCHP control process.
```

---

## Figure 4

### Working Title

`Why dynamic GT O&M allocation penalizes intermittent operation more strongly than conventional hour-based allocation`

### Intended Location

Section 3, near the per-step operating cost and the discussion of GT switching-sensitive burden.

### Main Message

This figure is intentionally analogous to the reference paper's Fig. 2, but it should be adapted to your own wording.
Its job is to explain, visually and honestly, why a cycling-sensitive O&M term matters.

The safest version is conceptual rather than fully numeric.
If later you want, you can replace symbolic cost labels with real numbers from your final configuration.

### Mandatory Visual Elements

- Two operating patterns:
  - continuous GT operation
  - intermittent GT operation with frequent starts
- Two costing rules compared side by side:
  - conventional hour-based O&M allocation
  - proposed dynamic / cycle-sensitive O&M allocation
- Timeline-style bars for GT on/off status
- Start events marked explicitly
- Cost accumulation shown underneath each timeline
- A final comparison note:
  - continuous operation: similar total O&M under both methods
  - intermittent operation: dynamic method yields higher total O&M because starts are charged more frequently

### Important Accuracy Note

Do not claim exact costs unless you later replace symbolic terms with verified values.
At the ideation stage, use symbolic labels like:

- `c_hr`
- `c_cycle`
- `N_start`
- `Total O&M = c_hr * operating hours + c_cycle * starts`

### Recommended Caption

`Illustration of the cycle-sensitive GT O&M allocation used in this study. Compared with a purely hour-based treatment, the dynamic formulation assigns additional burden to frequent start-stop behavior, thereby distinguishing continuous from intermittent GT operation more clearly.`

### Direct English Prompt

```text
Create a rigorous scientific presentation comparison figure that explains a dynamic gas-turbine O&M cost allocation idea for energy dispatch modeling. The figure should feel like a high-quality methods slide for a research talk, not like a business infographic. It must compare conventional hour-based O&M allocation with a proposed dynamic cycle-sensitive allocation, and it must clearly show why intermittent GT operation becomes more expensive when switching burden is explicitly charged. Use a clean vector-like academic style, white background, charcoal text, restrained outlines, and the Okabe-Ito palette with blue #0072B2 as the structural accent, orange #E69F00 for GT operating periods, and vermillion #D55E00 for start-stop penalty markers.

OVERALL LAYOUT:
Use a 2 by 2 comparison matrix.
Columns:
- left column: "Conventional hour-based O&M"
- right column: "Dynamic cycle-sensitive O&M"
Rows:
- top row: "Continuous GT operation"
- bottom row: "Intermittent GT operation"

Inside each of the four panels, draw a clean timeline with evenly spaced time ticks. Use a horizontal operating bar where orange segments indicate GT on-state and pale grey gaps indicate off-state. For the intermittent cases, show several separate orange operating segments. Mark every start event with a small vermillion triangle or vertical event marker labeled "start". Optionally place a small clean gas-turbine sketch icon near the panel title to reinforce that the timeline belongs to GT operation.

TOP LEFT PANEL:
continuous GT operation under conventional hour-based allocation. Show one long uninterrupted operating bar. Beneath it show a simple cost expression such as "O&M approx c_hr x operating hours". Add a small note: "no explicit cycling burden".

TOP RIGHT PANEL:
continuous GT operation under dynamic cycle-sensitive allocation. Show the same long uninterrupted operating bar, but add only one start marker near the beginning and a cost expression such as "O&M approx c_hr x operating hours + c_cycle x 1". Add a compact note saying the total remains close to the conventional case when operation is mostly continuous.

BOTTOM LEFT PANEL:
intermittent GT operation under conventional hour-based allocation. Show multiple separated on-periods but use the same hour-based cost expression "O&M approx c_hr x operating hours". Add a note indicating that frequent starts are not distinguished strongly.

BOTTOM RIGHT PANEL:
intermittent GT operation under dynamic cycle-sensitive allocation. Show the same fragmented operating profile, but add a start marker at the beginning of each on-period. Under the timeline write "O&M approx c_hr x operating hours + c_cycle x N_start". Add a visual emphasis that the accumulated cost is substantially higher because repeated starts are counted explicitly.

At the far right or bottom include a compact summary box with two bullet statements:
- "Continuous operation: both methods give similar O&M totals"
- "Intermittent operation: dynamic allocation penalizes frequent GT cycling"

Avoid fake realism, turbines as shiny machines, or generic finance imagery. This must look like an honest scientific methods figure with symbolic, teachable labels that can later be replaced by exact numerical factors if needed. Make the figure precise, calm, professional, presentation-ready, and publication-oriented.
```

---

## Practical Workflow With ChatGPT Image Generation

Recommended usage:

1. Start with the prompt of one figure only.
2. Ask ChatGPT to generate a first composition draft.
3. Then ask for one revision pass focused only on:
   - layout clarity
   - label density
   - less decoration
   - more academic honesty
4. After you get a useful composition, manually redraw the final version.

Useful follow-up instructions for ChatGPT or Nano Banana:

- `Make the figure more like a scientific PPT slide and less like a journal line drawing.`
- `Use recognizable facility sketch icons for GT, HRSG, boiler, TES, BES, ABS, and ECH.`
- `Keep the composition professional, not commercial, not cartoon.`
- `Make the labels shorter and more scientific.`
- `Keep the layout but simplify the visual clutter.`
- `Replace generic boxes for equipment with clean physical device silhouettes.`
- `Make the safety projection block more central and more readable.`

## Final Advice

For the final paper, the most valuable use of AI is:

- rapid composition ideation
- module arrangement inspiration
- color and layout exploration

The least valuable use is:

- trusting the generated text labels as publication-ready
- using the raw AI image directly without redrawing

The strongest final workflow is:

`AI composition draft -> manual scientific redraw -> LaTeX insertion`
