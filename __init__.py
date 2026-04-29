from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .nodes.fens_token_counter import FensTokenCounter
from .nodes.opti_empty_latent import OptiEmptyLatent
from .nodes.opti_empty_latent_advanced import OptiEmptyLatentAdvanced

WEB_DIRECTORY = "./web"


class FensSimpleNodesExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [FensTokenCounter, OptiEmptyLatent, OptiEmptyLatentAdvanced]


async def comfy_entrypoint() -> FensSimpleNodesExtension:
    return FensSimpleNodesExtension()


__all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]
