from typing import Dict, Type


class ComponentRegistry:
    def __init__(self):
        self.components: Dict[str, Dict[str, Type["PipelineStage"]]] = {}

    def register(self, stage_type: str, version: str):
        def decorator(cls):
            if stage_type not in self.components:
                self.components[stage_type] = {}
            self.components[stage_type][version] = cls
            return cls
        return decorator

    def get(self, stage_type: str, version: str) -> Type["PipelineStage"]:
        """ Get the class for a given stage type and version """
        if stage_type not in self.components.keys():
            raise ValueError(f"Unknown stage type: {stage_type}")

        stage_dict = self.components[stage_type]
        if version not in stage_dict.keys():
            raise ValueError(f"Unknown version for stage {stage_type}: {version}")

        return stage_dict[version]

registry = ComponentRegistry()
