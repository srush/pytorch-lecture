import torch
from torchtyping import TensorType as TT
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import integers, lists, composite, floats
from hypothesis import given
import numpy as np
import random
import sys
import typing

tensor = torch.tensor

numpy_to_torch_dtype_dict = {
    bool: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
}
torch_to_numpy_dtype_dict = {v: k for k, v in numpy_to_torch_dtype_dict.items()}


@composite
def spec(draw, x, min_size=1):
    # Get the type hints.
    if sys.version_info >= (3, 9):
        gth = typing.get_type_hints(x, include_extras=True)
    else:
        gth = typing.get_type_hints(x)

    # Collect all the dimension names.
    names = set()
    for k in gth:
        if not hasattr(gth[k], "__metadata__"):
            continue
        dims = gth[k].__metadata__[0]["details"][0].dims
        names.update([d.name for d in dims if isinstance(d.name, str)])
    names = list(names)

    # draw sizes for each dim.
    size = integers(min_value=min_size, max_value=5)
    arr = draw(arrays(shape=(len(names),), unique=True, elements=size, dtype=np.int32)).tolist()
    sizes = dict(zip(names, arr))
    for n in list(sizes.keys()):
        if '*' in n or '+' in n or '-' in n or '//' in n:
            i, op, j = n.split()
            i_val = i if i.isdigit() else sizes[i]
            j_val = j if j.isdigit() else sizes[j]
            sizes[n] = eval('{}{}{}'.format(i_val, op,j_val))

    # Create tensors for each size.
    ret = {}
    for k in gth:
        if not hasattr(gth[k], "__metadata__"):
            continue
        shape = tuple(
            [
                sizes[d.name] if isinstance(d.name, str) else d.size
                for d in gth[k].__metadata__[0]["details"][0].dims
            ]
        )
        dtype = (torch_to_numpy_dtype_dict[
                    gth[k].__metadata__[0]["details"][1].dtype
                ]
                if len(gth[k].__metadata__[0]["details"]) >= 2
                else int)
        ret[k] = draw(
            arrays(
                shape=shape,
                dtype=dtype,
                elements=integers(min_value=-5, max_value=5) if 
                         dtype == int else None,
                unique=False
            )
        )
        ret[k] = np.nan_to_num(ret[k], nan=0, neginf=0, posinf=0)
    ret["return"][:] = 0
    return ret, sizes


def make_test(name, problem, problem_spec, add_sizes=[], constraint=lambda d, sizes: d):
    examples = []
    for i in range(3):
        example, sizes = spec(problem, 3).example()
        example = constraint(example, sizes=sizes)
        out = example["return"].tolist()
        del example["return"]
        problem_spec(*example.values(), out)
        
        for size in add_sizes:
            example[size] = sizes[size]

        yours = None
        try:
            yours = problem(*map(tensor, example.values()))
            
        except AssertionError:
            pass
        for size in add_sizes:
            del example[size]
        example["target"] = tensor(out)
        if yours is not None:
            example["yours"] = yours 
        examples.append(example)
        
    @given(spec(problem))
    def test_problem(d):
        d, sizes = d

        d = constraint(d, sizes=sizes)
        out = d["return"].tolist()
        del d["return"]
        problem_spec(*d.values(), out)
        for size in add_sizes:
            d[size] = sizes[size]

        out2 = problem(*map(tensor, d.values()))
        out = tensor(out)
        out2 = torch.broadcast_to(out2, out.shape)
        assert torch.allclose(
            out, out2
        ), "Two tensors are not equal\n Spec: \n\t%s \n\t%s" % (out, out2)

    return test_problem


def run_test(fn):
    fn()
    # Generate a random puppy video if you are correct.
    print("Correct!")
    from IPython.display import HTML
    pups = [
    "2m78jPG",
    "pn1e9TO",
    "MQCIwzT",
    "udLK6FS",
    "ZNem5o3",
    "DS2IZ6K",
    "aydRUz8",
    "MVUdQYK",
    "kLvno0p",
    "wScLiVz",
    "Z0TII8i",
    "F1SChho",
    "9hRi2jN",
    "lvzRF3W",
    "fqHxOGI",
    "1xeUYme",
    "6tVqKyM",
    "CCxZ6Wr",
    "lMW0OPQ",
    "wHVpHVG",
    "Wj2PGRl",
    "HlaTE8H",
    "k5jALH0",
    "3V37Hqr",
    "Eq2uMTA",
    "Vy9JShx",
    "g9I2ZmK",
    "Nu4RH7f",
    "sWp0Dqd",
    "bRKfspn",
    "qawCMl5",
    "2F6j2B4",
    "fiJxCVA",
    "pCAIlxD",
    "zJx2skh",
    "2Gdl1u7",
    "aJJAY4c",
    "ros6RLC",
    "DKLBJh7",
    "eyxH0Wc",
    "rJEkEw4"]
    return HTML("""
    <video alt="test" controls autoplay=1 height="240">
        <source src="https://openpuppies.com/mp4/%s.mp4"  type="video/mp4">
    </video>
    """%(random.sample(pups, 1)[0]))
