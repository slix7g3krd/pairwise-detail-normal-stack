import pytest
import graphics.detail_normal_stack as mod

CASES = [
    ("tangent_space_unify", (1,)),
    ("normal_mip_renorm", (1,)),
    ("distance_weight", (1,)),
    ("tangent_basis_sync", (1,)),
    ("channel_convention", (1,)),
    ("decode_check", (1,)),
    ("load_material", (1,)),
    ("sample_detail", (1,)),
    ("compose_normal", (1,)),
    ("present_surface", (1,)),
    ("validate_material", (1,)),
    ("metrics_material", (1,)),
    ("audit_material", (1,)),
    ("fallback_material", (1,)),
    ("determinism_material", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
