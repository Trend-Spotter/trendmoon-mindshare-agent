# ------------------------------------------------------------------------------
#
#   Copyright 2025 xiuxiuxar
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------

"""Performance Tracker for Pearl v1 compliance."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from time import time
from typing import Optional, List, Any


@dataclass
class PerformanceMetric:
    """
    Represents a single performance metric (KPI).

    Attributes:
        name: Metric name (e.g., "Portfolio Value", "ROI")
        is_primary: Whether this is the primary metric
        description: Optional tooltip description (HTML allowed)
        value: The metric value as a string (e.g., "$1,234", "12.5%")
    """

    name: str
    is_primary: bool
    value: str
    description: Optional[str] = None


@dataclass
class AgentPerformance:
    """
    Complete agent performance data structure.

    Attributes:
        timestamp: UNIX timestamp of last update (UTC, in seconds)
        metrics: List of up to 2 metrics (primary and secondary)
        agent_behavior: Description of current agent behavior
    """

    timestamp: Optional[int]
    metrics: List[PerformanceMetric]
    agent_behavior: Optional[str]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "metrics": [asdict(m) for m in self.metrics],
            "agent_behavior": self.agent_behavior,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentPerformance":
        """Create from dictionary."""
        metrics = [PerformanceMetric(**m) for m in data.get("metrics", [])]
        return cls(
            timestamp=data.get("timestamp"),
            metrics=metrics,
            agent_behavior=data.get("agent_behavior"),
        )

    @classmethod
    def empty(cls) -> "AgentPerformance":
        """Create an empty performance record."""
        return cls(timestamp=None, metrics=[], agent_behavior=None)


class PerformanceTracker:
    """
    Manages agent performance tracking for Pearl v1.

    Handles reading, updating, and persisting the agent_performance.json file.
    """

    def __init__(self, context: Any, store_path: Optional[str] = None, auto_timestamp: bool = True):
        """
        Initialize the performance tracker.

        Args:
            context: Agent context for logging
            store_path: Path to store directory
            auto_timestamp: Automatically update timestamp on changes
        """
        self.context = context
        self.store_path = Path(store_path or "./persistent_data")
        self.file_path = self.store_path / "agent_performance.json"
        self.auto_timestamp = auto_timestamp

        # Ensure store path exists
        self.store_path.mkdir(parents=True, exist_ok=True)

        # Initialize file if it doesn't exist
        if not self.file_path.exists():
            self._save(AgentPerformance.empty())

    def _load(self) -> AgentPerformance:
        """Load performance data from file."""
        try:
            with open(self.file_path, "r") as f:
                data = json.load(f)
            return AgentPerformance.from_dict(data)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self.context.logger.warning(f"Failed to load performance data: {e}")
            return AgentPerformance.empty()

    def _save(self, performance: AgentPerformance) -> None:
        """Save performance data to file."""
        try:
            with open(self.file_path, "w") as f:
                json.dump(performance.to_dict(), f, indent=2)
        except Exception as e:
            self.context.logger.error(f"Failed to save performance data: {e}")

    def update_metric(
        self,
        name: str,
        value: str,
        is_primary: bool = False,
        description: Optional[str] = None,
        replace: bool = True,
    ) -> None:
        """
        Update or add a performance metric.

        Args:
            name: Metric name
            value: Metric value as string
            is_primary: Whether this is the primary metric
            description: Optional description (HTML allowed)
            replace: If True, replace existing metric with same name
        """
        performance = self._load()

        # Create new metric
        new_metric = PerformanceMetric(
            name=name,
            is_primary=is_primary,
            value=value,
            description=description,
        )

        # Find and replace existing metric, or add new
        replaced = False
        if replace:
            for i, metric in enumerate(performance.metrics):
                if metric.name == name:
                    performance.metrics[i] = new_metric
                    replaced = True
                    break

        if not replaced:
            performance.metrics.append(new_metric)

        # Ensure max 2 metrics
        if len(performance.metrics) > 2:
            # Keep primary and most recent
            primary = [m for m in performance.metrics if m.is_primary]
            secondary = [m for m in performance.metrics if not m.is_primary]
            performance.metrics = primary[:1] + secondary[-1:]

        # Update timestamp if auto mode
        if self.auto_timestamp:
            performance.timestamp = int(time())

        self._save(performance)

    def update_behavior(self, behavior: str) -> None:
        """
        Update the agent behavior description.

        Args:
            behavior: Description of current agent behavior
        """
        performance = self._load()
        performance.agent_behavior = behavior

        if self.auto_timestamp:
            performance.timestamp = int(time())

        self._save(performance)

    def reset(self) -> None:
        """Reset to empty state."""
        self._save(AgentPerformance.empty())
