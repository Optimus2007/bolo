# Myna landing page: director's sheet

THE SITE WHERE YOU: watch one Hinglish sentence fly into a myna and come out as something your laptop does,
then boss the page around in Hinglish yourself.
WORLD / MATERIAL: a voice. One live "voice line" (spring-chain waveform) runs through the whole page; it
wobbles with scroll speed and pointer, becomes the sentence, the divider, the road the bird flies on.
MOTION: living material = the voice line (1-D spring chain on canvas) · 5-second wow = a flat line in the dark
jumps when the word "bolo" is spoken in type, Myna's eye opens and the room irises open from it ·
subject motion: sentence draws as a wave, Myna flies along it and swallows it, wave bars scramble into
letters, words drop into slots, a pin drops on the real folder, a terminal types, 20 commands orbit Myna,
a cloud tries to steal the audio and gets pecked away, the page obeys typed commands ·
input -> material: pointer drags the voice line; scroll velocity makes it shout.
HERO SIGNATURE MOVE: the sentence's journey (pinned, scrubbed): wave -> bird -> words -> intent -> folder -> action.
CONTINUITY DEVICE: the voice line + Myna (fixed actor that changes verb per scene: wakes, listens, flies,
swallows, thinks, perches, orbits, pecks, sleeps).
JOURNEY: open (dark) -> hero (dusk room) -> the sentence -> inside the ear (Bolo Sun) -> words to meaning
(Bolo Tiny) -> the folder map (grounding) -> the laptop acts -> 20 commands ring -> the cloud that gets
nothing (privacy) -> honest numbers -> playground (climax: you command the page) -> dawn close (rename her).
OPENING RITUAL: flat line, "bolo" typed in Roman then Devanagari, the line jumps, an eye opens, iris.
CLOSING MOMENT: dawn; Myna asleep on the laptop; type a new name and the wordmark becomes it.
TYPE: Baloo 2 (display; Indian rounded letterforms, Devanagari built in) + Mukta (text; Indian, Devanagari).
TYPE REASON: hand-painted Indian shop signage, and the sentences are literally Hinglish in two scripts.
PALETTE: chapter light (one evening in a hostel room): dusk marigold -> night laptop glow -> dawn.
  sampled: marigold garland orange, turmeric yellow (myna's beak), chai brown, terracotta wall, neem green,
  laptop-screen teal (Myna's body), dusk sky blue-grey, tubelight white. Myna teal + turmeric are the actor;
  the ground drifts warm -> deep blue-green night -> pale dawn. No violet, no neon.
PERSONALITY LAYER: Myna's eyes follow the pointer; Myna replies in Hinglish in the playground.
VISUAL SOURCE: authored layered SVG (Myna, room, laptop, folder map, cloud), canvas voice line, CSS 3D ring,
kinetic type. GSAP 3.12.5 + ScrollTrigger + MotionPath from cdnjs; Google Fonts.
ASSUMPTIONS: install command and GitHub link are placeholders (repo is private, nothing published);
playground is a toy in the browser (keyword rules), clearly labelled; numbers are the real measured ones.

## Review (independent reviewer, round 1) and what changed
Round 1 score: about 6.0/10 (design 6, creativity 6, usability 5.5, content 7): "a competent website with animation on
top", short of the reference reels (8). Fixed after it: Myna now pecks the waveform, lands on the Downloads tile
(instead of a pin), zooms out through the laptop screen into the ring; less dead scroll; the wave line no longer
crosses text on phones; brand ("meet Myna") in the hero; bigger folder labels; label crossfade. Not re-scored
(self-assessed about 6.5). Next fixes: break the "headline top-left" template in 3 middle scenes (giant ghost 80,
full-screen cloud chase, playground centred on the bird); a vertical card stack for the ring on phones.
Run: `python3 -m http.server -d site 8088` then open http://127.0.0.1:8088/.
