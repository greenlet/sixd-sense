from pathlib import Path
from typing import List, Tuple

from pydantic import BaseModel, validator
from pydantic_yaml import YamlStrEnum, YamlModel
import yaml


class MyEnum(YamlStrEnum):
    """This is a custom enumeration that is YAML-safe."""

    a = "a"
    b = "b"

class InnerModel(BaseModel):
    """This is a normal pydantic model that can be used as an inner class."""

    fld: float = 1.0

class MyModel(YamlModel):
    """This is our custom class, with a `.yaml()` method.

    The `parse_raw()` and `parse_file()` methods are also updated to be able to
    handle `content_type='application/yaml'`, `protocol="yaml"` and file names
    ending with `.yml`/`.yaml`
    """

    x: int = 1
    e: MyEnum = MyEnum.a
    m: InnerModel = InnerModel()

    @validator('x')
    def _chk_x(cls, v: int) -> int:  # noqa
        """You can add your normal pydantic validators, like this one."""
        assert v > 0
        return v

m1 = MyModel(x=2, e="b", m=InnerModel(fld=1.5))

# This dumps to YAML and JSON respectively
yml = m1.yaml()
jsn = m1.json()
print(type(yml))

with open('temp.yaml', 'w') as f:
    f.write(yml)

m2 = MyModel.parse_raw(yml)  # This automatically assumes YAML
assert m1 == m2

m3 = MyModel.parse_raw(jsn)  # This will fallback to JSON
assert m1 == m3

m4 = MyModel.parse_raw(yml, proto="yaml")
assert m1 == m4

m5 = MyModel.parse_raw(yml, content_type="application/yaml")
assert m1 == m5


class Config(YamlModel):
    ds_path: Path
    # x: List[int]
    x: Tuple[int, int]
    y: List[str]

with open('dsgen_config_debug.yaml', 'r') as f:
    cfg = Config.parse_raw(f.read())
print(cfg)
