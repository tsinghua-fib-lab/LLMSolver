from rl4co.envs import RL4COEnvBase
from torch import nn

model_type_list = ['mtpomo', 'mvmoe', 'rf-pomo', 'rf-moe', 'rf-transformer', ]


def get_policy(model_type: str):
    if model_type == 'mtpomo':
        try:
            from solver.cvrp.mtpomo_solver import MTPOMOPolicy
            return MTPOMOPolicy()
        except ImportError:
            return None
    elif model_type == 'mvmoe':
        try:
            from solver.cvrp.mvmoe_solver.policy import MVMoEPolicy
            return MVMoEPolicy(hierarchical_gating=False)
        except ImportError:
            return None
    elif model_type == 'rf-pomo':
        try:
            from solver.cvrp.routefinder_solver.policy import RouteFinderPolicy
            return RouteFinderPolicy()
        except ImportError:
            return None
    elif model_type == 'rf-moe':
        try:
            from solver.cvrp.mvmoe_solver.policy import MVMoEPolicy
            from solver.cvrp.routefinder_solver.env_embeddings.mtvrp.init import MTVRPInitEmbeddingRouteFinder
            from solver.cvrp.routefinder_solver.env_embeddings.mtvrp.context import MTVRPContextEmbeddingRouteFinder
            embed_dim = 128
            init_embedding = MTVRPInitEmbeddingRouteFinder(embed_dim=embed_dim)
            context_embedding = MTVRPContextEmbeddingRouteFinder(embed_dim=embed_dim)
            return MVMoEPolicy(embed_dim=embed_dim,
                               init_embedding=init_embedding,
                               context_embedding=context_embedding,
                               hierarchical_gating=False)
        except ImportError:
            return None
    elif model_type == 'rf-transformer':
        try:
            from solver.cvrp.routefinder_solver import RouteFinderPolicy
            return RouteFinderPolicy(normalization="rms",
                                     encoder_use_prenorm=True,
                                     encoder_use_post_layers_norm=True,
                                     parallel_gated_kwargs={
                                         "mlp_activation": "silu"
                                     })
        except ImportError:
            return None
    return None

def get_model(model_type: str, env: RL4COEnvBase, policy: nn.Module):
    if model_type == 'mtpomo':
        try:
            from solver.cvrp.mtpomo_solver import MTPOMO
            return MTPOMO(env=env, policy=policy, normalize_reward="none")
        except ImportError:
            return None
    elif model_type == 'mvmoe':
        try:
            from solver.cvrp.mvmoe_solver import MVMoE
            if env.generator.subsample is not None:
                env.generator.subsample = None
                print(f"Warning: the env.generator.subsample is set to None")
            return MVMoE(env=env, policy=policy, normalize_reward="none")
        except ImportError:
            return None
    elif model_type == 'rf-pomo':
        try:
            from solver.cvrp.routefinder_solver import RouteFinderBase
            return RouteFinderBase(env=env, policy=policy)
        except Exception as e:
            print(e)
            return None
    elif model_type == 'rf-moe':
        try:
            from solver.cvrp.routefinder_solver.model import RouteFinderMoE
            return RouteFinderMoE(env=env, policy=policy)
        except ImportError:
            return None
    elif model_type == 'rf-transformer':
        try:
            from solver.cvrp.routefinder_solver import RouteFinderBase
            return RouteFinderBase(env=env, policy=policy)
        except ImportError:
            return None
