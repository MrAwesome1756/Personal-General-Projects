# NEYMAR JR vs THE END — 60s cinematic cut

A 60-second vertical (1080×1920, 30fps) Minecraft-style mini-movie, rendered entirely from code:
pixel-art rigs, parallax camera, particles, post-FX, and a fully synthesized score + sound design
(no samples, no licensed music).

Output: `output/neymar_vs_the_end_60s.mp4`

## Story (boss-fight arc)

| Time | Shot | Transition in |
|---|---|---|
| 0–8s | Cold open: "THE END. / NO ONE HAS EVER SCORED HERE." → tilt down through obsidian pillars; dragon silhouette crosses the far sky | fade from black |
| 8–10s | Portal close-up: 14 Eyes of Ender click in, portal ignites | flash cut |
| 10–14s | Low angle: Neymar materialises, walks out, spins the ball — **NEYMAR JR / THE No.10** | white flash |
| 14–16s | Darkness: Enderman eyes open one pair at a time | glitch cut |
| 16–19s | Tracking shot, dribbling on the beat; Enderman teleports in | zoom blur |
| 19–20s | Extreme close-up: Enderman jaw drops, screech | hard cut + shake |
| 20–21s | Close-up: Neymar smirks at the camera | whip pan |
| 21–24.5s | Speed-ramped crossover (slow-mo through the legs) → freeze frame **ANKLES BROKEN** | whip pan |
| 24.5–27s | VHS rewind + mirrored low-angle half-speed replay | glitch/rewind |
| 27–29s | Ground shakes, End crystals fire beams into the sky | dip from black |
| 29–32s | Looking up: Ender Dragon swoops over the camera — **FINAL BOSS** | vertical whip |
| 32–34s | Dragon lands, shockwave, roar | wing wipe |
| 34–35s | Dragon eye ECU, pupil narrows | punch-in |
| 35–36s | Low-angle hero shot, ball spin, crouch | punch-in |
| 36–40s | Drop: sprint, spin move past the bite, runs up the dragon, launches | flash on the drop |
| 40–43s | Slow-mo mid-air tomahawk, camera roll, heartbeat | speed ramp |
| 43–45s | **POSTERIZED** — backboard shatters, rim bends, dragon slumps | impact flash |
| 45–47s | Two instant replays (under-the-rim, wide mirrored) | flash cuts |
| 47–51s | Dragon rises, light bursts out, explodes into particles + XP | flash |
| 51–54s | Endermen crowd celebrates, confetti | Minecraft block dissolve |
| 54–60s | Victory poster with the Dragon Egg trophy → "THE END... WAS JUST THE BEGINNING." | flash / fade |

## Re-rendering

```bash
pip install numpy scipy opencv-python-headless pillow imageio-ffmpeg
cd src
python audio.py ../output/score.wav
python render.py video ../output/neymar_vs_the_end_60s.mp4 --audio ../output/score.wav
python render.py sheet 1 sheet.png          # 1-fps contact sheet for review
python render.py still 43.3 still.png       # single frame
```

## Code map

- `src/core.py` — camera, textured-quad blitting, text styles, particles, post-FX (bloom, grade, chroma, glitch, VHS, keystone, motion/zoom blur, letterbox)
- `src/rigs.py` — pixel textures + rigs: Neymar (side/front), Enderman (side/front), Ender Dragon, ball
- `src/world.py` — sky/nebula/stars, obsidian pillars + End crystals, island, hoop, End portal
- `src/shots.py` — the shot list, choreography, cameras, transitions
- `src/audio.py` — synthesized score (120 BPM, D minor → D major) and all SFX, synced to the shot timings
- `src/render.py` — parallel renderer → ffmpeg

Fonts: Anton, Bebas Neue, Press Start 2P (SIL Open Font License).
