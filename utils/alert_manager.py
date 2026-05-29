"""
alert_manager.py
────────────────
Manages the teacher-absent alert escalation chain:
  Level 1 (60 s)  → notify teacher
  Level 2 (120 s) → notify coordinator / principal
  Level 3 (180 s) → escalate (log / alarm)
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    NONE = 0
    TEACHER = 1        # initial reminder
    COORDINATOR = 2    # escalation
    CRITICAL = 3       # full escalation


@dataclass
class AlertEvent:
    level: AlertLevel
    timestamp: float
    chaos_score: float
    message: str
    acknowledged: bool = False


class AlertManager:
    """
    Stateful alert machine.

    Usage:
        mgr = AlertManager(teacher_threshold=60, coord_threshold=120)
        event = mgr.update(teacher_present=False, chaos_score=75.0)
        if event:
            send_notification(event)
    """

    def __init__(
        self,
        chaos_trigger: float = 50.0,
        teacher_threshold: int = 60,
        coord_threshold: int = 120,
        critical_threshold: int = 180,
        cooldown: int = 300,
    ):
        self.chaos_trigger = chaos_trigger
        self.teacher_threshold = teacher_threshold
        self.coord_threshold = coord_threshold
        self.critical_threshold = critical_threshold
        self.cooldown = cooldown

        self._absence_start: Optional[float] = None
        self._last_alert_time: float = 0.0
        self._current_level = AlertLevel.NONE
        self.event_log: List[AlertEvent] = []

    # ── public API ────────────────────────────────────────────────────────
    def update(
        self,
        teacher_present: bool,
        chaos_score: float,
    ) -> Optional[AlertEvent]:
        now = time.time()

        if teacher_present or chaos_score < self.chaos_trigger:
            self._reset()
            return None

        # start absence timer
        if self._absence_start is None:
            self._absence_start = now

        elapsed = now - self._absence_start

        # determine target level
        if elapsed >= self.critical_threshold:
            target = AlertLevel.CRITICAL
        elif elapsed >= self.coord_threshold:
            target = AlertLevel.COORDINATOR
        elif elapsed >= self.teacher_threshold:
            target = AlertLevel.TEACHER
        else:
            return None

        # only fire if we've moved to a higher level
        if target.value <= self._current_level.value:
            return None

        # cooldown between same-level pings
        if now - self._last_alert_time < self.cooldown and \
                target == self._current_level:
            return None

        self._current_level = target
        self._last_alert_time = now

        event = AlertEvent(
            level=target,
            timestamp=now,
            chaos_score=chaos_score,
            message=self._build_message(target, elapsed, chaos_score),
        )
        self.event_log.append(event)
        logger.warning(f"ALERT [{target.name}]: {event.message}")
        return event

    def acknowledge(self, index: int = -1):
        if self.event_log:
            self.event_log[index].acknowledged = True
        self._reset()

    def get_elapsed_absence(self) -> float:
        if self._absence_start is None:
            return 0.0
        return time.time() - self._absence_start

    def get_current_level(self) -> AlertLevel:
        return self._current_level

    # ── helpers ───────────────────────────────────────────────────────────
    def _reset(self):
        self._absence_start = None
        self._current_level = AlertLevel.NONE

    @staticmethod
    def _build_message(level: AlertLevel, elapsed: float, chaos: float) -> str:
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        elapsed_str = f"{mins}m {secs}s"
        chaos_str = f"{chaos:.0f}%"

        if level == AlertLevel.TEACHER:
            return (f"⚠️ Teacher Reminder: Your classroom has been unmonitored for "
                    f"{elapsed_str}. Chaos level: {chaos_str}. Please return or respond.")
        elif level == AlertLevel.COORDINATOR:
            return (f"🔴 Coordinator Alert: Classroom unmonitored for {elapsed_str}. "
                    f"Teacher has not responded. Chaos level: {chaos_str}. "
                    f"Immediate action required.")
        else:
            return (f"🚨 CRITICAL ESCALATION: Classroom unmonitored for {elapsed_str}. "
                    f"Chaos level: {chaos_str}. Escalating to Principal.")