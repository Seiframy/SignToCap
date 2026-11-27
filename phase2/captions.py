"""Initial streaming policy shared by replay and the planned cloud service."""

from dataclasses import dataclass


def activity_features(keypoints, scores, fps):
    import numpy as np

    joints = [7, 8, 9, 10, *range(91, 133)]
    anchor = (keypoints[:, 0, 5] + keypoints[:, 0, 6]) / 2
    shoulder_span = np.linalg.norm(keypoints[:, 0, 5] - keypoints[:, 0, 6], axis=-1)
    position = (keypoints[:, 0, joints] - anchor[:, None]) / np.maximum(shoulder_span[:, None, None], 0.05)
    confidence = scores[:, 0, joints]
    shoulders_visible = np.min(scores[:, 0, [5, 6]], axis=1) > 0.3
    visible = (confidence > 0.3) & shoulders_visible[:, None]
    motion = np.zeros(len(keypoints), dtype=np.float32)
    distance = np.linalg.norm(np.diff(position, axis=0), axis=-1) * fps
    distance[~(visible[1:] & visible[:-1])] = 0
    # Average the fastest visible joints to catch either hand.
    motion[1:] = np.sort(distance, axis=-1)[:, -5:].mean(axis=1)
    return motion, visible.mean(axis=1)


def boundary_features(motion, visibility, end, fps):
    recent = slice(max(0, end - round(0.25 * fps)), end)
    previous = slice(max(0, end - round(0.85 * fps)), max(1, end - round(0.25 * fps)))
    return [
        float(motion[recent].mean()),
        float(motion[recent].max()),
        float(motion[previous].mean()),
        float(visibility[recent].mean()),
    ]


@dataclass
class CaptionState:
    previous: str = ""
    repeats: int = 0

    def update(self, text, *, quiet, armed, learned_boundary, punctuation, overflow):
        if text == self.previous and text:
            self.repeats += 1
        else:
            self.repeats = int(bool(text))
        previous, current = self.previous.split(), text.split()
        common = 0
        for old, new in zip(previous, current):
            if old != new:
                break
            common += 1
        self.previous = text
        stable = " ".join(current[:common])
        evidence = quiet >= 0.65 or (punctuation and quiet >= 0.35 and text.endswith((".", "?", "!")))
        if learned_boundary is not None:
            evidence = quiet >= 0.20 and learned_boundary
        if armed and evidence and self.repeats >= 2:
            return "final", stable
        if overflow:
            return "uncertain", stable
        return "partial", stable


class StreamingSession:
    """One bounded, serial pose stream. Supply monotonically ordered fixed-cadence chunks."""

    def __init__(self, options, translator, fps, boundary_head=None):
        if fps is None or not 0 < fps <= 30:
            raise ValueError("Replay requires a verified constant frame rate at or below 30 fps")
        self.options = options
        self.translator = translator
        self.fps = fps
        self.boundary_head = boundary_head
        self.cap = min(256, round(options.buffer_s * fps))
        self.seen = 0
        self.first = self.last_active = None
        self.keypoints = self.scores = self.motion = self.visible = None
        self.state = CaptionState()

    def push(self, keypoints, scores, *, decode=True):
        import numpy as np

        if not len(keypoints) or len(keypoints) != len(scores):
            raise ValueError("Expected matching nonempty pose chunks")
        points = keypoints
        confidence = scores
        if self.keypoints is not None:
            points = np.concatenate((self.keypoints[-1:], keypoints))
            confidence = np.concatenate((self.scores[-1:], scores))
        motion, visible = activity_features(points, confidence, self.fps)
        motion, visible = motion[-len(keypoints):], visible[-len(keypoints):]
        active = np.flatnonzero((motion >= self.options.motion) & (visible >= 0.2))
        if len(active):
            if self.first is None:
                self.first = max(0, self.seen + int(active[0]) - round(0.25 * self.fps))
            self.last_active = self.seen + int(active[-1])
        self.seen += len(keypoints)
        for attr, values in (("keypoints", keypoints), ("scores", scores), ("motion", motion), ("visible", visible)):
            previous = getattr(self, attr)
            combined = values if previous is None else np.concatenate((previous, values))
            setattr(self, attr, combined[-self.cap:].copy())
        return self.decode() if decode else None

    def decode(self):
        if self.first is None:
            return None
        length = min(self.seen - self.first, self.cap)
        if length < max(2, round(0.5 * self.fps)):
            return None
        text, _, timing = self.translator.translate(self.keypoints[-length:], self.scores[-length:],
                                                    beams=self.options.beams, tokens=self.options.tokens)
        quiet = (self.seen - self.last_active - 1) / self.fps
        boundary = None
        if self.boundary_head is not None:
            boundary = self.boundary_head(boundary_features(self.motion, self.visible, len(self.motion), self.fps))
        kind, stable = self.state.update(
            text, quiet=quiet, armed=self.visible[-length:].mean() >= 0.2,
            learned_boundary=boundary, punctuation=self.options.policy == "stable-punctuation",
            overflow=self.seen - self.first > self.cap,
        )
        event = {"kind": kind, "text": text, "stable_prefix": stable, "source_end_ms": self.seen / self.fps * 1000,
                 "segment_start_ms": self.first / self.fps * 1000,
                 "inferred_end_ms": (self.last_active + 1) / self.fps * 1000, **timing}
        if kind != "partial":
            self.first = self.last_active = None
            self.state = CaptionState()
        return event

    def finish(self):
        if self.first is None:
            return None
        event = {"kind": "uncertain", "reason": "end_of_input", "text": self.state.previous,
                 "segment_start_ms": self.first / self.fps * 1000}
        self.first = self.last_active = None
        self.state = CaptionState()
        return event


class AdaptiveStreamingSession(StreamingSession):
    """Throttle preview decoding, display agreement, and expose an explicit full-window commit."""

    def __init__(self, options, translator, fps, boundary_head=None):
        super().__init__(options, translator, fps, boundary_head)
        self.last_preview = None
        self.displayed = ""
        self.interval = max(1, round(options.decode_interval_ms * fps / 1000))

    def reset_caption(self):
        self.first = self.last_active = self.last_preview = None
        self.displayed = ""
        self.state = CaptionState()

    def event(self, kind, text="", **metadata):
        return {"kind": kind, "text": text, "source_end_ms": self.seen * 1000 / self.fps,
                "segment_start_ms": None if self.first is None else self.first * 1000 / self.fps, **metadata}

    def decode(self):
        if self.first is None:
            return None
        length = self.seen - self.first
        if length > self.cap:
            result = self.event("uncertain", reason="buffer_limit")
            self.reset_caption()
            return result
        if length < max(2, round(0.5 * self.fps)):
            return None
        quiet = (self.seen - self.last_active - 1) / self.fps
        if (self.options.policy != "explicit" and self.state.repeats >= 2
                and self.visible[-length:].mean() >= 0.2
                and (quiet >= 0.65 or (self.options.policy == "stable-punctuation"
                     and quiet >= 0.35 and self.state.previous.endswith((".", "?", "!"))))):
            return self.commit(origin="automatic_pause")
        if self.last_preview is not None and self.seen - self.last_preview < self.interval:
            return None
        self.last_preview = self.seen
        text, _, timing = self.translator.translate(
            self.keypoints[-length:], self.scores[-length:], beams=1, tokens=self.options.partial_tokens,
        )
        if repeated_phrase(text):
            self.state = CaptionState()
            return self.event("telemetry", reason="repetitive_preview", **timing)
        _, agreed = self.state.update(text, quiet=0, armed=False, learned_boundary=None,
                                      punctuation=False, overflow=False)
        if not agreed or agreed == self.displayed:
            return self.event("telemetry", reason="unstable_or_unchanged", **timing)
        self.displayed = agreed
        return self.event("partial", agreed, stable_prefix=agreed, **timing)

    def commit(self, *, origin="operator"):
        length = 0 if self.first is None else self.seen - self.first
        if length < max(2, round(0.5 * self.fps)) or length > self.cap:
            result = self.event("uncertain", reason="no_complete_window", commit_origin=origin)
        else:
            text, _, timing = self.translator.translate(self.keypoints[-length:], self.scores[-length:],
                                                        beams=self.options.beams, tokens=self.options.tokens)
            text = text.strip()
            enough_pose = self.visible[-length:].mean() >= 0.2
            if text and enough_pose and not repeated_phrase(text):
                result = self.event("final", text, commit_origin=origin, **timing)
            else:
                result = self.event("uncertain", reason="repetition_or_insufficient_evidence",
                                    commit_origin=origin, **timing)
        self.reset_caption()
        return result


def repeated_phrase(text):
    words = [word.strip(".,;:!?").casefold() for word in text.split()]
    for width in range(1, min(12, len(words) // 3) + 1):
        for start in range(len(words) - 3 * width + 1):
            span = words[start:start + 3 * width]
            if span == span[:width] * 3:
                return True
    return False
