import json
import os
from datetime import datetime
from typing import Optional

from autotune.fc.pid import PIDProfile
from autotune.fc.rate import RateProfile


class ProfileManager:
    def __init__(self, profiles_dir: str = "profiles"):
        self.profiles_dir = profiles_dir
        os.makedirs(profiles_dir, exist_ok=True)

    def save_profile(
        self,
        name: str,
        pid_profile: PIDProfile,
        rate_profile: RateProfile,
        notes: str = "",
    ) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.json"
        filepath = os.path.join(self.profiles_dir, filename)

        data = {
            "name": name,
            "timestamp": timestamp,
            "notes": notes,
            "pid": pid_profile.to_dict(),
            "rate": rate_profile.to_dict(),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return filepath

    def load_profile(self, filepath: str) -> Optional[dict]:
        if not os.path.exists(filepath):
            return None

        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_profiles(self) -> list[dict]:
        profiles = []
        if not os.path.exists(self.profiles_dir):
            return profiles

        for filename in os.listdir(self.profiles_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.profiles_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        data["filepath"] = filepath
                        data["filename"] = filename
                        profiles.append(data)
                except Exception:
                    continue

        profiles.sort(key=lambda p: p.get("timestamp", ""), reverse=True)
        return profiles

    def delete_profile(self, filepath: str) -> bool:
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False

    def compare_profiles(self, filepath1: str, filepath2: str) -> dict:
        p1 = self.load_profile(filepath1)
        p2 = self.load_profile(filepath2)

        if not p1 or not p2:
            return {}

        comparison = {"profile1": p1.get("name", "Profile 1"),
                      "profile2": p2.get("name", "Profile 2"),
                      "pid": {}, "rate": {}}

        pid1 = p1.get("pid", {})
        pid2 = p2.get("pid", {})

        for axis in ["Roll", "Pitch", "Yaw"]:
            a1 = pid1.get(axis, {})
            a2 = pid2.get(axis, {})
            comparison["pid"][axis] = {}
            for key in ["P", "I", "D"]:
                v1 = a1.get(key, 0)
                v2 = a2.get(key, 0)
                diff = v2 - v1 if abs(v1) > 0.01 else 0
                comparison["pid"][axis][key] = {"profile1": v1, "profile2": v2, "diff": diff}

        rate1 = p1.get("rate", {})
        rate2 = p2.get("rate", {})
        for axis in ["Roll", "Pitch", "Yaw"]:
            a1 = rate1.get(axis, {})
            a2 = rate2.get(axis, {})
            comparison["rate"][axis] = {}
            for key in ["RC_Rate", "Super_Rate", "RC_Expo"]:
                v1 = a1.get(key, 0)
                v2 = a2.get(key, 0)
                diff = v2 - v1
                comparison["rate"][axis][key] = {"profile1": v1, "profile2": v2, "diff": diff}

        return comparison