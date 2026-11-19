"""Copy the source documents from their origin repositories into data/raw."""

import shutil

from rag_chatbot.config.Settings import Settings
from rag_chatbot.config.SourceCatalog import SourceCatalog


class SyncSourcesCommand:
    """data/raw is ignored by git, so a fresh clone fills it from the sibling repositories first."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def run(self) -> int:
        catalog = SourceCatalog.load(self.settings.sources_file, self.settings.root)
        missing = 0
        for spec in catalog.sources:
            if spec.origin is None or not spec.origin.is_file():
                print(f"  missing origin for {spec.name}: {spec.origin}")
                missing += 1
                continue
            spec.path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(spec.origin, spec.path)
            print(f"  copied {spec.name}: {spec.origin} -> {spec.path.relative_to(self.settings.root)}")
        return 1 if missing else 0
