from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .nodes.FensTokenCounter import FensTokenCounter
from .nodes.OptiEmptyLatent import OptiEmptyLatent
from .nodes.OptiEmptyLatentAdvanced import OptiEmptyLatentAdvanced

WEB_DIRECTORY = "./web"


class FensSimpleNodesExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [FensTokenCounter, OptiEmptyLatent, OptiEmptyLatentAdvanced]


async def comfy_entrypoint() -> FensSimpleNodesExtension:
    return FensSimpleNodesExtension()


__all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]
