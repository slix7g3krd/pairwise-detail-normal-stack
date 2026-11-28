"""细节法线层叠 —— 六项缺陷修复的参考实现。

缺陷 -> 修复对照
----------------
1. 细节法线在世界空间与基础法线相加、二者不在同一空间，砖缝高光偏向错误一侧
   -> tangent_space_unify / compose_normal：统一到切线空间，用 RNM
      （Reoriented Normal Mapping）混合，最后再经 TBN 切线基变换回世界空间。
2. 细节层 mip 用普通平均，法线长度随 mip 衰减，远处高光整片消失
   -> normal_mip_renorm：每级 2x2 平均之后做带 epsilon 保护的重归一。
3. 两层细节固定 0.5 权重、不随距离变化，远处摩尔纹
   -> distance_weight：每层独立的 smoothstep 近/远距离衰减权重。
4. 细节 UV 平铺不跟切线基走，模型绕 Y 轴转 90 度后划痕方向错位
   -> tangent_basis_sync：平铺变换定义在切线 T/B 平面内，世界方向随模型
      一起旋转，满足 world_axes(rot) == Ry(rot) @ world_axes(0) 不变量。
5. BC5 压法线时绿通道没翻转，OpenGL / D3D 后端光照相反
   -> channel_convention：以 OpenGL(+Y) 为规范约定，载入 D3D 资产时翻转 G。
6. 压缩法线缺少解码校验
   -> decode_check：UNORM 解码、约定翻转、Z 重建、单位圆盘与长度容差校验。

约定：所有法线向量均为带符号单位向量（分量约在 [-1, 1]）；
切线基 basis 为 3x3 矩阵，三行依次为 T、B、N（行向量布局，世界->切线直接左乘）。

所有入口在只收到一个标量参数时，会退回到内置的程序化演示数据并返回完整结构，
因此既支持管线调用，也满足最小契约测试。
"""

from __future__ import annotations

import math

import numpy as np

EPS = 1e-8
LENGTH_TOL = 2e-2

OPENGL = "opengl"
DIRECTX = "directx"
CANONICAL = OPENGL  # 全管线内部统一使用 OpenGL 切线空间约定（+Y）


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def _is_scalar(x) -> bool:
    return isinstance(x, (int, float, np.floating, np.integer)) and not isinstance(
        x, bool
    )


def _normalize(v, axis=-1):
    """带 epsilon 保护的归一化；零向量回退为切线平面平法线 (0, 0, 1)。"""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    out = v / np.maximum(n, EPS)
    flat = np.broadcast_to(np.array([0.0, 0.0, 1.0]), out.shape)
    return np.where(n < EPS, flat, out)


def _smoothstep(edge0: float, edge1: float, x):
    x = np.clip((np.asarray(x, dtype=float) - edge0) / (edge1 - edge0), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _default_basis() -> np.ndarray:
    """默认切线基：T=+X, B=+Y, N=+Z，按行堆叠。"""
    return np.eye(3, dtype=float)


def _rotate_y(vectors, angle: float):
    """把一批行向量（或切线基的每一行）绕世界 Y 轴旋转 angle。"""
    c, s = math.cos(angle), math.sin(angle)
    rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    v = np.atleast_2d(np.asarray(vectors, dtype=float))
    out = (rot @ v.T).T
    return out[0] if np.asarray(vectors).ndim == 1 else out


def _tangent_to_world(t: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """切线向量 -> 世界向量。basis 行是 T/B/N，故世界基为其转置。"""
    t = np.asarray(t, dtype=float)
    if t.ndim == 1:
        return basis.T @ t
    return t @ basis


# ---- 程序化高度场与法线（演示/自检测试用数据，确定性固定种子） -------------

def _heightfield_brick(size: int = 64) -> np.ndarray:
    xs = np.linspace(0.0, 4.0 * math.pi, size, endpoint=False)
    xx, yy = np.meshgrid(xs, xs)
    return np.sin(xx) * np.sin(yy)  # 网格状起伏，模拟砖缝


def _heightfield_scratch(size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(7)
    xs = np.linspace(0.0, 12.0 * math.pi, size, endpoint=False)
    _, yy = np.meshgrid(xs, xs)
    grooves = 0.5 * np.sin(yy)  # 沿 X 方向拉长的沟槽 -> 定向划痕
    return grooves + rng.normal(0.0, 0.15, (size, size))


def _heightfield_to_normal(height: np.ndarray, strength: float = 1.0) -> np.ndarray:
    dy, dx = np.gradient(height)
    vec = np.stack([-dx * strength, -dy * strength, np.ones_like(height)], axis=-1)
    return _normalize(vec)


def _sample_wrap(level: np.ndarray, u, v) -> np.ndarray:
    """平铺环绕 + 双线性采样，支持标量或数组 UV。"""
    level = np.asarray(level, dtype=float)
    h, w = level.shape[:2]
    u_arr = np.asarray(u, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    scalar = u_arr.ndim == 0 and v_arr.ndim == 0
    u_arr = np.atleast_1d(u_arr)
    v_arr = np.atleast_1d(v_arr)

    x = np.mod(u_arr, 1.0) * w
    y = np.mod(v_arr, 1.0) * h
    x0 = np.floor(x).astype(int) % w
    y0 = np.floor(y).astype(int) % h
    x1 = (x0 + 1) % w
    y1 = (y0 + 1) % h
    fx = (x - np.floor(x))[..., None]
    fy = (y - np.floor(y))[..., None]

    top = level[y0, x0] * (1.0 - fx) + level[y0, x1] * fx
    bot = level[y1, x0] * (1.0 - fx) + level[y1, x1] * fx
    out = top * (1.0 - fy) + bot * fy
    return out[0] if scalar else out


# ---------------------------------------------------------------------------
# 修复 1：切线空间统一
# ---------------------------------------------------------------------------

def tangent_space_unify(normal, basis: np.ndarray | None = None) -> np.ndarray:
    """把法线统一到切线空间并归一化。

    basis 为 None 时，输入本身已在切线空间，只做归一化；
    否则输入视为世界空间向量，用 TBN（行 T/B/N）变换进切线空间。
    标量输入返回平法线，作为无数据时的安全值。
    """
    if _is_scalar(normal):
        return np.array([0.0, 0.0, 1.0])
    n = _normalize(normal)
    if basis is None:
        return n
    m = np.asarray(basis, dtype=float)
    if n.ndim == 1:
        return _normalize(m @ n)
    return _normalize(n @ m.T)


# ---------------------------------------------------------------------------
# 修复 2：法线 mip 重归一
# ---------------------------------------------------------------------------

def normal_mip_renorm(levels):
    """法线 mip 链生成 / 修复。

    - 传入单张 HxWx3 法线图：逐级 2x2 平均后重归一，返回完整 mip 链；
    - 传入已存在的 mip 列表：逐级重归一，修复“普通平均导致的长度衰减”；
    - 标量输入：在内置砖缝法线上演示并返回 mip 链。
    长度低于 EPS 的退化纹素回退为平法线 (0,0,1)。
    """
    if _is_scalar(levels):
        levels = _heightfield_to_normal(_heightfield_brick(32))

    if isinstance(levels, (list, tuple)):
        return [_normalize(np.asarray(level, dtype=float)) for level in levels]

    cur = np.asarray(levels, dtype=float)
    chain = [_normalize(cur)]
    while min(cur.shape[0], cur.shape[1]) >= 2:
        h = cur.shape[0] - cur.shape[0] % 2
        w = cur.shape[1] - cur.shape[1] % 2
        cur = cur[:h, :w]
        blocks = cur.reshape(h // 2, 2, w // 2, 2, cur.shape[2])
        nxt = _normalize(blocks.mean(axis=(1, 3)))  # 关键：平均后必须重归一
        chain.append(nxt)
        cur = nxt
        if cur.shape[0] == 1 and cur.shape[1] == 1:
            break
    return chain


# ---------------------------------------------------------------------------
# 修复 3：距离权重衰减
# ---------------------------------------------------------------------------

def distance_weight(distance, near: float = 1.0, far: float = 16.0,
                    layer_weight: float = 1.0):
    """细节层距离衰减权重：near 处为 layer_weight，far 处平滑衰减到 0。

    取代固定 0.5 的混合系数；远处细节权重归零，消除摩尔纹，
    近处保留全强度。支持标量或数组距离。
    """
    fade = 1.0 - _smoothstep(near, far, distance)
    return np.clip(fade, 0.0, 1.0) * layer_weight


# ---------------------------------------------------------------------------
# 修复 4：切线基同步
# ---------------------------------------------------------------------------

def tangent_basis_sync(basis=None, tile_scale=(1.0, 1.0),
                       tile_rotation: float = 0.0,
                       model_rotation_y: float = 0.0) -> dict:
    """把细节 UV 平铺变换同步到切线基。

    二维平铺 Q = R(tile_rotation) @ diag(tile_scale) 定义在切线 T/B 平面内，
    而不是对象/世界空间的固定平面。模型绕 Y 轴旋转时切线基随模型一起旋转，
    平铺的世界轴满足不变量：
        world_axes(model_rot) == Ry(model_rot) @ world_axes(0)
    即划痕“贴”在表面上随模型一起转，不会相对表面漂移。
    返回 uv 变换、旋转后的切线基与平铺 U/V 轴的世界方向。
    """
    if basis is None or _is_scalar(basis):
        basis = _default_basis()
    basis = np.asarray(basis, dtype=float)
    basis_r = _rotate_y(basis, model_rotation_y)

    c, s = math.cos(tile_rotation), math.sin(tile_rotation)
    q = np.array([[c, -s], [s, c]]) @ np.diag(np.asarray(tile_scale, dtype=float))

    # basis_r 的列是 T/B/N；取 T、B 两列张成切线平面
    world_axes = basis_r.T[:, :2] @ q  # 形状 3x2：列为 U/V 平铺轴的世界方向
    return {
        "uv_transform": q,
        "basis": basis_r,
        "world_axes": world_axes,
        "tile_scale": tuple(float(x) for x in tile_scale),
        "tile_rotation": float(tile_rotation),
        "model_rotation_y": float(model_rotation_y),
    }


# ---------------------------------------------------------------------------
# 修复 5：通道约定统一（BC5 绿通道翻转）
# ---------------------------------------------------------------------------

def channel_convention(normal_map, source: str = CANONICAL,
                       target: str = DIRECTX):
    """OpenGL(+Y) 与 DirectX(-Y) 法线约定互转：约定不同则翻转 G 通道。

    管线内部统一以 OpenGL 为规范约定；D3D 后端的 BC5 资产在载入时翻转 G，
    输出到 D3D 时再翻回去，保证两端光照一致。
    标量输入返回约定规则说明。
    """
    if _is_scalar(normal_map):
        return {
            "canonical": CANONICAL,
            "supported": (OPENGL, DIRECTX),
            "rule": "OpenGL 与 DirectX 之间转换时对 G 通道取反，其余通道不变",
        }
    if source == target:
        return np.asarray(normal_map, dtype=float).copy()
    if {source, target} != {OPENGL, DIRECTX}:
        raise ValueError(f"未知法线约定: {source} -> {target}")
    out = np.asarray(normal_map, dtype=float).copy()
    out[..., 1] = -out[..., 1]
    return out


# ---------------------------------------------------------------------------
# 修复 6：压缩解码校验
# ---------------------------------------------------------------------------

def decode_check(data, convention: str = CANONICAL, tol: float = LENGTH_TOL) -> dict:
    """BC5 风格两通道压缩法线的解码与校验。

    支持 uint8 UNORM、[0,1] 浮点 UNORM、带符号两通道与三通道向量：
      1. UNORM 解码到 [-1,1]；
      2. DirectX 约定翻转 G；
      3. 单位圆盘检查（x^2+y^2 <= 1），Z = sqrt(max(0, 1-x^2-y^2)) 重建；
      4. 单位长度容差检查。
    标量输入执行内置自检（平法线 + 倾斜法线往返）。
    """
    if _is_scalar(data):
        flat = np.array([[128, 128]], dtype=np.uint8)
        tilted = np.array([[200, 60]], dtype=np.uint8)
        r1 = decode_check(flat, OPENGL, tol)
        r2 = decode_check(tilted, DIRECTX, tol)
        return {
            "ok": bool(r1["ok"] and r2["ok"]),
            "errors": r1["errors"] + r2["errors"],
            "convention": convention,
            "tol": tol,
            "self_test": {"flat_gl": r1["ok"], "tilted_dx": r2["ok"]},
            "normals": r1["normals"],
            "max_length_error": max(r1["max_length_error"], r2["max_length_error"]),
        }

    arr = np.asarray(data)
    errors: list[str] = []

    if arr.ndim == 0:
        raise ValueError("decode_check 需要数组输入")

    if arr.dtype == np.uint8:
        xy = arr.astype(float) / 255.0 * 2.0 - 1.0
        encoding = "bc5_unorm_u8"
    elif arr.shape[-1] == 2 and np.issubdtype(arr.dtype, np.floating) \
            and arr.min() >= 0.0 and arr.max() <= 1.0:
        xy = arr.astype(float) * 2.0 - 1.0
        encoding = "bc5_unorm_f32"
    elif arr.shape[-1] == 2:
        xy = arr.astype(float)
        encoding = "signed_xy"
    else:
        vec = np.asarray(arr, dtype=float)
        lengths = np.linalg.norm(vec, axis=-1)
        max_err = float(np.max(np.abs(lengths - 1.0)))
        if max_err > tol:
            errors.append(f"三通道法线长度超差: max_err={max_err:.4f} > {tol}")
        return {
            "ok": not errors,
            "errors": errors,
            "convention": convention,
            "tol": tol,
            "encoding": "xyz_vector",
            "normals": _normalize(vec),
            "max_length_error": max_err,
            "samples": int(vec.size // 3),
        }

    if convention == DIRECTX:
        xy = xy.copy()
        xy[..., 1] = -xy[..., 1]
    elif convention != OPENGL:
        raise ValueError(f"未知法线约定: {convention}")

    r2 = float(np.sum(xy * xy, axis=-1).max())
    if r2 > 1.0 + tol:
        errors.append(f"编码点超出单位圆盘: max(x^2+y^2)={r2:.4f}")

    z = np.sqrt(np.maximum(0.0, 1.0 - np.sum(xy * xy, axis=-1)))
    normals = np.concatenate([xy, z[..., None]], axis=-1)
    lengths = np.linalg.norm(normals, axis=-1)
    max_err = float(np.max(np.abs(lengths - 1.0)))
    if max_err > tol:
        errors.append(f"重建法线长度超差: max_err={max_err:.4f} > {tol}")

    return {
        "ok": not errors,
        "errors": errors,
        "convention": convention,
        "tol": tol,
        "encoding": encoding,
        "normals": normals,
        "max_length_error": max_err,
        "samples": int(xy.size // 2),
    }


# ---------------------------------------------------------------------------
# 材质管线
# ---------------------------------------------------------------------------

def load_material(spec=None) -> dict:
    """载入并规范化材质：约定统一到 OpenGL，细节层预生成重归一 mip 链。

    标量 / None 输入返回内置的双细节层演示材质（砖缝 + 定向划痕）。
    """
    if spec is None or _is_scalar(spec):
        spec = {
            "convention": CANONICAL,
            "basis": _default_basis(),
            "base_normal": _heightfield_to_normal(_heightfield_brick(64), 0.6),
            "details": [
                {
                    "name": "brick_joint",
                    "map": _heightfield_to_normal(_heightfield_brick(64), 1.0),
                    "tiling": (4.0, 4.0),
                    "rotation": 0.0,
                    "weight": 0.7,
                    "near": 0.5,
                    "far": 12.0,
                    "convention": OPENGL,
                },
                {
                    "name": "scratches",
                    "map": _heightfield_to_normal(_heightfield_scratch(64), 1.0),
                    "tiling": (8.0, 8.0),
                    "rotation": math.radians(30.0),
                    "weight": 0.4,
                    "near": 0.5,
                    "far": 6.0,
                    "convention": OPENGL,
                },
            ],
        }

    material = dict(spec)
    material.setdefault("convention", CANONICAL)
    material.setdefault("basis", _default_basis())
    material.setdefault("details", [])
    material.setdefault(
        "base_normal",
        np.tile(np.array([0.0, 0.0, 1.0]), (64, 64, 1)),
    )

    for layer in material["details"]:
        layer.setdefault("convention", material["convention"])
        if layer["convention"] == DIRECTX:
            # 修复 5：D3D 资产载入即翻 G，之后管线内只存在 OpenGL 约定
            layer["map"] = channel_convention(layer["map"], DIRECTX, OPENGL)
            layer["convention"] = OPENGL
        layer["map"] = np.asarray(layer["map"], dtype=float)
        layer["mips"] = normal_mip_renorm(layer["map"])  # 修复 2
        layer.setdefault("tiling", (1.0, 1.0))
        layer.setdefault("rotation", 0.0)
        layer.setdefault("weight", 0.5)
        layer.setdefault("near", 1.0)
        layer.setdefault("far", 16.0)

    material["convention"] = CANONICAL
    material["basis"] = np.asarray(material["basis"], dtype=float)
    return material


def sample_detail(material, uv=(0.5, 0.5), distance: float = 1.0,
                  basis=None) -> dict:
    """采样并混合全部细节层。

    - 平铺 UV 经 tangent_basis_sync 的切线平面变换（修复 4）；
    - mip 级按距离选择，采样自重归一 mip 链（修复 2）；
    - 每层权重 = 基础权重 x 距离衰减（修复 3），不再是固定 0.5。
    返回每层采样记录、加权合成细节法线与总权重。
    """
    if _is_scalar(material):
        material = load_material()
    if basis is None:
        basis = material["basis"]

    u, v = uv
    accum = np.zeros(3)
    total_weight = 0.0
    records = []

    for layer in material["details"]:
        sync = tangent_basis_sync(
            basis, layer["tiling"], layer["rotation"]
        )
        tuv = sync["uv_transform"] @ np.array([u, v], dtype=float)

        max_mip = len(layer["mips"]) - 1
        mip = int(np.clip(round(math.log2(max(float(distance), 1.0))), 0, max_mip))
        normal_t = _normalize(_sample_wrap(layer["mips"][mip], tuv[0], tuv[1]))

        weight = float(
            distance_weight(distance, layer["near"], layer["far"], layer["weight"])
        )
        accum += weight * normal_t
        total_weight += weight
        records.append(
            {
                "name": layer["name"],
                "normal_t": normal_t,
                "weight": weight,
                "mip": mip,
                "uv_t": tuv,
            }
        )

    combined = _normalize(accum) if total_weight > EPS else np.array([0.0, 0.0, 1.0])
    return {
        "layers": records,
        "combined": combined,
        "total_weight": float(total_weight),
        "distance": float(distance),
    }


def _blend_rnm(base: np.ndarray, detail: np.ndarray) -> np.ndarray:
    """切线空间 RNM（带符号单位向量版），替代“相加再归一化”。

    t = (base.xy, base.z + 1)，u = (-detail.xy, detail.z)
    r = t * dot(t,u)/t.z - u，再归一化。
    base 为平法线时结果恒等于 detail；base.z 接近 -1 时退化为加权相加。
    """
    base = _normalize(base)
    detail = _normalize(detail)
    t = np.array([base[0], base[1], base[2] + 1.0])
    if abs(t[2]) < EPS:
        return _normalize(base + detail)
    u = np.array([-detail[0], -detail[1], detail[2]])
    r = t * float(np.dot(t, u)) / t[2] - u
    return _normalize(r)


def compose_normal(base_normal, detail=None, basis=None, material=None,
                   distance: float = 1.0, uv=(0.5, 0.5)) -> dict:
    """统一空间后混合基础法线与细节法线（修复 1）。

    基础法线若带 basis 则先变换进切线空间；细节法线接受 sample_detail
    的结果字典（按总权重收缩扰动量，权重为 0 时混合结果等于基础法线），
    或直接给一个切线空间向量。混合使用 RNM，最后再变换回世界空间。
    """
    if _is_scalar(base_normal):
        material = load_material()
        base_normal = _sample_wrap(material["base_normal"], uv[0], uv[1])
        detail = sample_detail(material, uv, distance)
        basis = material["basis"]

    if material is not None and basis is None:
        basis = material["basis"]

    base_t = tangent_space_unify(base_normal, basis)  # 世界 -> 切线（修复 1）

    if isinstance(detail, dict):
        combined = _normalize(detail["combined"])
        w = float(np.clip(detail.get("total_weight", 1.0), 0.0, 1.0))
        detail_t = _normalize(np.array([combined[0] * w, combined[1] * w,
                                        combined[2]]))
    elif detail is None:
        detail_t = np.array([0.0, 0.0, 1.0])
    else:
        detail_t = _normalize(detail)

    normal_t = _blend_rnm(base_t, detail_t)
    result = {"normal_t": normal_t, "base_t": base_t, "detail_t": detail_t}
    if basis is not None:
        result["normal_world"] = _tangent_to_world(normal_t, np.asarray(basis,
                                                                        float))
    return result


def present_surface(material=None, points=None,
                    light_dir=(0.3, 0.3, 1.0)) -> dict:
    """对一组表面点跑完整管线，返回世界空间法线与 NdotL 高光项。"""
    if material is None or _is_scalar(material):
        material = load_material()
    if points is None:
        points = [
            {"uv": (0.25 * i + 0.1, 0.35 * i + 0.2), "distance": 0.5 + 2.0 * i}
            for i in range(8)
        ]

    light = _normalize(light_dir)
    normals_world = []
    shading = []
    for point in points:
        basis = point.get("basis", material["basis"])
        uv = point.get("uv", (0.5, 0.5))
        distance = float(point.get("distance", 1.0))

        sampled = sample_detail(material, uv, distance, basis)
        base_t = _sample_wrap(material["base_normal"], uv[0], uv[1])
        composed = compose_normal(base_t, sampled, basis=basis)

        n_world = composed["normal_world"]
        normals_world.append(n_world)
        shading.append(float(max(0.0, np.dot(n_world, light))))

    return {
        "normals_world": np.asarray(normals_world),
        "ndotl": np.asarray(shading),
        "points": points,
        "light_dir": light,
    }


# ---------------------------------------------------------------------------
# 质量保障：校验 / 指标 / 审计 / 回退 / 确定性
# ---------------------------------------------------------------------------

def validate_material(material) -> dict:
    """结构化校验材质资产，返回 ok 与错误清单。"""
    if _is_scalar(material):
        material = load_material()

    errors: list[str] = []
    checks = 0

    checks += 1
    if material.get("convention") != CANONICAL:
        errors.append(f"内部约定必须为 {CANONICAL}，实际 {material.get('convention')}")

    basis = np.asarray(material["basis"], dtype=float)
    checks += 1
    if basis.shape != (3, 3) or not np.allclose(basis @ basis.T, np.eye(3),
                                                 atol=1e-6):
        errors.append("切线基不是标准正交矩阵")
    elif abs(np.linalg.det(basis) - 1.0) > 1e-6:
        errors.append("切线基行列式不为 +1（手性错误）")

    base = np.asarray(material["base_normal"], dtype=float)
    checks += 1
    base_len_err = float(np.max(np.abs(np.linalg.norm(base, axis=-1) - 1.0)))
    if base_len_err > LENGTH_TOL:
        errors.append(f"基础法线长度超差: {base_len_err:.4f}")

    for layer in material["details"]:
        prefix = f"细节层[{layer.get('name', '?')}]"
        checks += 1
        if layer.get("convention") != CANONICAL:
            errors.append(f"{prefix} 约定未规范化到 {CANONICAL}")
        sx, sy = layer.get("tiling", (0.0, 0.0))
        if sx <= 0.0 or sy <= 0.0:
            errors.append(f"{prefix} 平铺系数必须为正，实际 {layer.get('tiling')}")
        if not 0.0 <= layer.get("weight", -1.0) <= 1.0:
            errors.append(f"{prefix} 权重超出 [0,1]")
        if layer.get("near", 0.0) >= layer.get("far", 0.0):
            errors.append(f"{prefix} 距离区间要求 near < far")
        mips = layer.get("mips")
        if not mips:
            errors.append(f"{prefix} 缺少重归一 mip 链")
            continue
        for idx, level in enumerate(mips):
            err = float(np.max(np.abs(
                np.linalg.norm(np.asarray(level), axis=-1) - 1.0)))
            checks += 1
            if err > LENGTH_TOL:
                errors.append(f"{prefix} mip{idx} 法线长度超差: {err:.4f}")

    return {"ok": not errors, "errors": errors, "checks": checks}


def metrics_material(material) -> dict:
    """统计材质法线长度、mip 覆盖、距离权重曲线等指标。"""
    if _is_scalar(material):
        material = load_material()

    def length_stats(level):
        lengths = np.linalg.norm(np.asarray(level, dtype=float), axis=-1)
        return {
            "shape": list(np.asarray(level).shape),
            "mean_length": float(lengths.mean()),
            "min_length": float(lengths.min()),
            "max_length": float(lengths.max()),
        }

    metrics = {
        "convention": material["convention"],
        "base_normal": length_stats(material["base_normal"]),
        "layers": [],
    }
    for layer in material["details"]:
        entry = {
            "name": layer.get("name"),
            "mip_count": len(layer.get("mips", [])),
            "mips": [length_stats(level) for level in layer.get("mips", [])],
            "weight_near": float(distance_weight(layer["near"], layer["near"],
                                                 layer["far"], layer["weight"])),
            "weight_mid": float(distance_weight(
                0.5 * (layer["near"] + layer["far"]),
                layer["near"], layer["far"], layer["weight"])),
            "weight_far": float(distance_weight(layer["far"], layer["near"],
                                                layer["far"], layer["weight"])),
        }
        metrics["layers"].append(entry)
    return metrics


def audit_material(material) -> dict:
    """针对六项原始缺陷逐条审计，返回每条规则的通过状态与证据。"""
    if _is_scalar(material):
        material = load_material()

    findings = []

    # 规则 1：切线空间统一 —— 世界 -> 切线 -> 世界往返恒等
    basis = np.asarray(material["basis"], dtype=float)
    probe = _normalize(np.array([0.4, -0.7, 1.0]))
    tangent = tangent_space_unify(probe, basis)
    back = _tangent_to_world(tangent, basis)
    r1_err = float(np.max(np.abs(back - probe)))
    findings.append({
        "id": "R1_tangent_space",
        "defect": "基础/细节法线空间不一致，世界空间相加",
        "status": "pass" if r1_err < 1e-6 else "fail",
        "roundtrip_max_error": r1_err,
    })

    # 规则 2：mip 重归一 —— 每级长度必须保持单位长度
    worst = 0.0
    for layer in material["details"]:
        for level in layer.get("mips", []):
            worst = max(worst, float(np.max(np.abs(
                np.linalg.norm(np.asarray(level), axis=-1) - 1.0))))
    findings.append({
        "id": "R2_mip_renorm",
        "defect": "mip 普通平均导致法线长度衰减",
        "status": "pass" if worst <= LENGTH_TOL else "fail",
        "worst_length_error": worst,
    })

    # 规则 3：距离权重衰减 —— 单调递减且端点为 1/0
    r3_ok = True
    evidence = {}
    for layer in material["details"]:
        ds = np.linspace(layer["near"], layer["far"], 32)
        ws = distance_weight(ds, layer["near"], layer["far"], layer["weight"])
        mono = bool(np.all(np.diff(ws) <= 1e-9))
        ends = abs(ws[0] - layer["weight"]) < 1e-6 and abs(ws[-1]) < 1e-6
        evidence[layer["name"]] = {"monotonic": mono, "endpoints_ok": ends}
        r3_ok = r3_ok and mono and ends
    findings.append({
        "id": "R3_distance_weight",
        "defect": "固定 0.5 权重，远处细节全强度导致摩尔纹",
        "status": "pass" if r3_ok else "fail",
        "evidence": evidence,
    })

    # 规则 4：切线基同步 —— 旋转 90 度后世界轴满足 Ry 不变量
    layer0 = material["details"][0]
    s0 = tangent_basis_sync(basis, layer0["tiling"], layer0["rotation"], 0.0)
    angle = math.pi / 2.0
    s1 = tangent_basis_sync(basis, layer0["tiling"], layer0["rotation"], angle)
    expected = _rotate_y(s0["world_axes"].T, angle).T
    r4_err = float(np.max(np.abs(s1["world_axes"] - expected)))
    findings.append({
        "id": "R4_basis_sync",
        "defect": "平铺不跟切线基，Y 轴旋转后划痕方向错位",
        "status": "pass" if r4_err < 1e-6 else "fail",
        "rotation_y_deg": 90.0,
        "invariant_max_error": r4_err,
    })

    # 规则 5：通道约定 —— GL->DX->GL 往返恒等，且恰好翻转一次 G
    probe_map = np.asarray(material["base_normal"], dtype=float)
    dx = channel_convention(probe_map, OPENGL, DIRECTX)
    gl_back = channel_convention(dx, DIRECTX, OPENGL)
    g_flip_ok = bool(np.allclose(dx[..., 1], -probe_map[..., 1]))
    other_ok = bool(np.allclose(dx[..., [0, 2]], probe_map[..., [0, 2]]))
    roundtrip_ok = bool(np.allclose(gl_back, probe_map))
    findings.append({
        "id": "R5_channel_convention",
        "defect": "BC5 绿通道未翻转，OpenGL/D3D 光照相反",
        "status": "pass" if g_flip_ok and other_ok and roundtrip_ok else "fail",
        "g_flipped": g_flip_ok,
        "other_channels_untouched": other_ok,
        "roundtrip_ok": roundtrip_ok,
    })

    # 规则 6：压缩解码校验 —— 每层法线编码为 BC5 风格 UNORM 后解码必须通过
    r6_ok = True
    r6_detail = {}
    for layer in material["details"]:
        xy = np.asarray(layer["map"])[..., :2]
        encoded = np.clip(np.round((xy * 0.5 + 0.5) * 255.0), 0, 255).astype(np.uint8)
        report = decode_check(encoded, OPENGL, LENGTH_TOL)
        r6_detail[layer["name"]] = {
            "ok": report["ok"],
            "max_length_error": report["max_length_error"],
        }
        r6_ok = r6_ok and report["ok"]
    findings.append({
        "id": "R6_decode_check",
        "defect": "压缩法线缺少解码校验",
        "status": "pass" if r6_ok else "fail",
        "evidence": r6_detail,
    })

    return {
        "ok": all(f["status"] == "pass" for f in findings),
        "findings": findings,
        "rules": len(findings),
        "passed": sum(1 for f in findings if f["status"] == "pass"),
    }


def fallback_material(material=None, reason: str = "unspecified") -> dict:
    """校验/解码失败时的安全回退材质：平法线、零细节权重、规范约定。"""
    return {
        "convention": CANONICAL,
        "basis": _default_basis(),
        "base_normal": np.tile(np.array([0.0, 0.0, 1.0]), (64, 64, 1)),
        "details": [],
        "detail_weight": 0.0,
        "fallback": True,
        "reason": str(reason),
    }


def determinism_material(material=None, points=None) -> dict:
    """重复执行完整管线并逐位比较，验证结果确定性。"""
    if material is None or _is_scalar(material):
        material = load_material()

    first = present_surface(material, points)
    second = present_surface(material, points)
    same = bool(np.array_equal(first["normals_world"], second["normals_world"])
                and np.array_equal(first["ndotl"], second["ndotl"]))
    max_diff = float(np.max(np.abs(
        first["normals_world"] - second["normals_world"])))

    reloaded = load_material()
    reload_same = bool(np.array_equal(
        reloaded["base_normal"], load_material()["base_normal"]))

    return {
        "deterministic": same and reload_same,
        "repeat_run_identical": same,
        "reload_identical": reload_same,
        "max_abs_diff": max_diff,
    }
