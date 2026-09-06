import json
import os


class SLUSchema:
    def __init__(self, path=None):
        self.data = {}
        if path and os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                self.data = json.load(f)

    def get_valid_domains(self):
        return self.data.get("domains", [])

    def get_valid_intents(self, domain):
        return self.data.get("domain2intents", {}).get(domain, [])

    def get_valid_slot_keys(self, domain, intent):
        return self.data.get("domain_intent2slot_keys", {}).get(f"{domain}|||{intent}", [])

    def get_valid_implicit_slot_keys(self, domain, intent):
        return self.data.get("domain_intent2implicit_slot_keys", {}).get(
            f"{domain}|||{intent}", []
        )

    def has_observed_surface_forms(self):
        forms = self.data.get("constraint_surface_forms", {})
        return isinstance(forms, dict) and bool(forms)

    def get_surface_forms(self):
        forms = self.data.get("constraint_surface_forms", {})
        if not isinstance(forms, dict):
            return {}
        return {
            str(name): tuple(str(value) for value in values)
            for name, values in forms.items()
            if isinstance(values, list)
        }
