"""Optimizer roles, declared by the modules that own the parameters.

Each hybrid module with parameters defines `optimizer()`, yielding
`(parameter, role)` for every parameter it creates, including those of its plain
`torch.nn` or shared children (Linear, RMSNorm, SwiGLU). `optim.parameter_groups`
collects the declarations by parameter identity and refuses any parameter that is
undeclared or declared twice.
"""

from torch import nn

ROLES = ("muon", "adamw_decay", "adamw_no_decay", "adamw_fast")


def declare(role, *owners):
    """`(parameter, role)` for each parameter, or for every parameter of each module."""
    if role not in ROLES:
        raise ValueError(f"Unknown optimizer role {role!r}")
    for owner in owners:
        for parameter in owner.parameters() if isinstance(owner, nn.Module) else (owner,):
            yield parameter, role
