"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.agilex_policy as agilex_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.calvin_policy as calvin_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# NOTE: 绕过 tyro 直接使用 nnx.filterlib.Filter 时的问题。
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # assets 目录；未提供时使用 config.assets_dirs，可用于从 base model checkpoint 等集中位置加载 assets。
    assets_dir: str | None = None

    # asset id；未提供时使用 repo id，用来引用不同机器人平台的 assets。
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id；None 表示创建 fake data。
    repo_id: str | None = None
    # assets 目录下存放 data assets 的子目录。
    asset_id: str | None = None
    # 预计算 normalization stats；None 表示不做 normalization。
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Data contract: 将 dataset 专用格式改写成 data_transforms 期望的通用格式。
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data contract: data_transforms 通常包含机器人专用转换，并在 normalization 之前执行。
    # 规范化后的字段契约见 `model.Observation` 和 `model.Actions`。
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # 模型专用 transforms，在数据 normalization 之后执行。
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # 为 True 时使用 quantile normalization，否则使用普通 z-score normalization。
    use_quantile_norm: bool = False

    # Data contract: data loader 用这些 key 生成 action sequence，长度由 model config 的 action_horizon 决定。
    # 若 LeRobot dataset 使用不同 action key，需要同步调整。
    action_sequence_keys: Sequence[str] = ("actions",)

    # 为 True 时使用 LeRobot dataset task 生成 prompt。
    prompt_from_task: bool = False

    # 仅供 RLDS data loader 使用，目前主要用于 DROID。
    rlds_data_dir: str | None = None
    # DROID dataset 的 action space。
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # Data contract: 采样的 dataset 列表包含 name、version、weight 和可选 filter_dict_path。
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""
    """
    已在 config.py (line 110) 的 ModelTransformFactory.__call__ 里加了中文标注。
    标注内容覆盖了三个选项：
        PI0：原版 pi0，默认 Paligemma prompt tokenizer，只做输入侧 transform。
        PI05：pi0.5 路径，Pi0FasterConfig / Pi0DiffusionConfig 也走这里。
        PI0_FAST：FAST tokenizer 路径，标明了输入编码和输出 action 解码两部分。
    """
    # Data contract: 输入样本没有 "prompt" 时注入该默认 prompt，保证后续 tokenizer 总能拿到语言条件。
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """根据 model_config.model_type 选择模型侧输入/输出 transforms。"""
        # model_type 是总开关：不同模型家族需要不同的 prompt/action 编码方式。
        match model_config.model_type:
            # 选项 1: PI0 原版模型。使用 Paligemma 文本 tokenizer，action horizon 使用默认 pad 逻辑。
            case _model.ModelType.PI0:
                # Group 把多个 transform 串成一条 pipeline；这里只定义模型输入侧 transforms。
                return _transforms.Group(
                    # inputs 中 transform 按顺序执行：prompt -> image -> token -> state/action shape。
                    inputs=[
                        # 如果样本没有 "prompt"，就写入 default_prompt；已有 prompt 时不覆盖。
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        # 把所有 image slot resize 到模型视觉塔期望的 224x224。
                        _transforms.ResizeImages(224, 224),
                        # 把 prompt 文本转成 Paligemma token，写入 tokenized_prompt / tokenized_prompt_mask。
                        _transforms.TokenizePrompt(
                            # max_token_len 来自模型配置，限制语言 token 的最大长度。
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        # 把 state/actions 的最后一维 pad 到模型 action_dim；PI0 这里不显式传 action_horizon。
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            # 选项 2: PI05 模型。FASTER / diffusion 配置也会走这个分支，因为它们复用 PI05 数据格式。
            case _model.ModelType.PI05:
                # 这个分支需要读取 Pi0Config/Pi0FasterConfig 特有字段，例如 discrete_state_input/action_horizon。
                assert isinstance(model_config, pi0_config.Pi0Config) or isinstance(
                    model_config, pi0_config.Pi0FasterConfig
                )
                # FASTER / diffusion: model_type 仍返回 PI05/PI0，因此复用常规 pi05/pi0 transform 路径。
                # 它们都返回连续 action chunk，不需要像 PI0_FAST 那样增加 action token 解码输出 transform。
                return _transforms.Group(
                    # inputs 中 transform 按顺序执行；FASTER 的 action_prefix 也会随这些输入 transform 一起处理。
                    inputs=[
                        # 补默认 prompt，保证语言条件一定存在。
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        # 统一图像大小，保证后续视觉 tokenizer 输入 shape 固定。
                        _transforms.ResizeImages(224, 224),
                        # 使用 Paligemma tokenizer；PI05 可选择是否把 state 离散化后拼进语言 token。
                        _transforms.TokenizePrompt(
                            # 构造文本 tokenizer，token 长度由模型配置控制。
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            # discrete_state_input=True 时，state 会以离散 token 形式进入 prompt/token 序列。
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        # 同时 pad action_dim 和 action_horizon；FASTER 需要固定 horizon 来处理 action_prefix/delay。
                        _transforms.PadStatesAndActions(model_config.action_dim, model_config.action_horizon),
                    ],
                )
            # 选项 3: PI0_FAST 模型。它可以理解为一条经过优化/替换的 action 生成路径：
            # 模型不直接返回连续 actions，而是自回归生成 FAST action tokens，再由 outputs 解码回连续动作。
            case _model.ModelType.PI0_FAST:
                # 若配置没有指定 tokenizer 类，就使用默认 FASTTokenizer。
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                # 若配置没有指定 tokenizer 参数，就传空 kwargs；否则透传自定义 tokenizer 参数。
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                # FAST pipeline 同时定义 inputs 和 outputs：输入要编码，输出要从 FAST token 解码回连续 action。
                # 这就是 PI0_FAST 和 PI0/PI05 最大的 transform 差异；普通模型 sample_actions 已经返回连续 actions。
                return _transforms.Group(
                    # inputs 负责把 observation/prompt 转成 FAST 模型可读的输入。
                    inputs=[
                        # 补默认 prompt，保持和其他模型家族一致的 prompt 契约。
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        # 视觉输入仍统一 resize 到 224x224。
                        _transforms.ResizeImages(224, 224),
                        # FAST 输入 tokenizer 会把 prompt 和动作相关上下文编码成 FAST 模型格式。
                        _transforms.TokenizeFASTInputs(
                            # 用选定 tokenizer 类和参数创建 tokenizer 实例。
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    # outputs 负责把模型输出 token 还原成环境可执行的连续 actions；
                    # 如果跳过这一步，后续 Unnormalize 会把离散 token 误当成归一化后的连续 action。
                    outputs=[
                        # 从 FAST 输出中提取 action chunk，并恢复到指定 horizon/dim。
                        _transforms.ExtractFASTActions(
                            # 输出解码必须使用和输入编码一致的 tokenizer 配置。
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            # action_horizon 决定一次输出多少步 action。
                            action_horizon=model_config.action_horizon,
                            # action_dim 决定每步 action 的维度。
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # LeRobot 数据集 repo id。
    repo_id: str = tyro.MISSING
    # 控制 assets 的加载方式。
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # factory 会在这个 base config 上做定制化更新。
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)  # asset_id 是绝对路径时会直接生效
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # data transforms 的 factory。
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # model transforms 的 factory。
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # Data contract: 为 True 时，送入模型前将 joint 维度转成相对当前 state 的 delta；gripper 保持 absolute。
    use_delta_joint_actions: bool = True
    # Data contract: 输入缺少 "prompt" 时注入该默认 prompt。
    default_prompt: str | None = None
    # Data contract: 为 True 时把标准 Aloha joint/gripper 值转换到 base model 训练使用的 pi internal runtime 空间。
    # 标准 Aloha 数据应保持为 True。
    adapt_to_pi: bool = True

    # Data contract: Aloha repack 把 LeRobot 键名对齐到 AlohaInputs 契约。
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Data contract: 从这些 dataset keys 读取 action sequence。
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # 3.2 LeRobotLiberoDataConfig.create：data_loader.create_data_loader 会调用到这里。
        # 对 `pi05_libero`，config.data 就是 LeRobotLiberoDataConfig，所以会进入当前 create。
        # 这个函数不直接读取样本；它只组装“之后每个样本要按什么顺序转换”的 DataConfig。

        # assets_dirs 是当前 config 的 assets 目录，例如 assets/pi05_libero。
        # create_base_config 会从这里加载 normalization stats，后面的 Normalize transform 会用这些统计量归一化 state/actions。
        # model_config 是模型配置；对 `pi05_libero`，它是 Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)。

        # 3.3 配置 repack_transforms：把 LeRobot dataset 原始 key 改成 LIBERO adapter 期望的 key。
        # Data contract: repack 只作用于离线 dataset，不作用于在线推理；它把 LeRobot 转换脚本产出的键名
        # 对齐到 LIBERO 推理 adapter 的输入契约。自定义 dataset 时，先看 policy server 实际传入哪些 key，
        # 再把这里的映射改成“dataset key -> 推理契约 key”。
        # 这里左边是转换后的统一 key，右边是 LeRobot dataset 原始 key；例如把原始 "image" 放到 "observation/image"。
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # 3.4 配置 data_transforms：把 LIBERO 格式转成 openpi 统一输入格式。
        # Data contract: data_transforms 同时用于训练和推理；inputs 把 LIBERO 契约转成 openpi 统一输入，
        # outputs 只在推理后把模型输出转回环境 action。若换自定义 adapter，就替换这里的 inputs/outputs。
        # LiberoInputs 会把 observation/image、observation/wrist_image、observation/state 转成：
        # state、image dict、image_mask dict、actions、prompt，这正是 Observation.from_dict 前需要的格式。
        # model_type 来自 model_config；pi05_libero 的 model_type 是 PI05，因此右腕占位图会被 mask 掉。
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # 3.4 的可选扩展：某些数据集若存的是 absolute action，可在这里补 delta action 转换。
        # Data contract: Pi0 训练通常使用相对当前 state 的 delta actions；若 dataset 存的是 absolute 关节目标，
        # 需要用 DeltaActions 转换 joint 维度并让 gripper 保持 absolute。LIBERO 原始 actions 已是 delta，
        # 只有兼容旧 Pi0 checkpoint 时才额外开启这里的 DeltaActions。
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # 3.5 配置 model_transforms：处理 prompt tokenization、图像尺寸、state/action padding 等模型侧格式。
        # Data contract: model_transforms 处理 prompt tokenization、图像尺寸、state/action padding 等模型侧格式；
        # 这层与具体 dataset key 无关，自定义数据集通常只改 repack/data_transforms。
        # 对 pi05_libero，这里会进入 ModelTransformFactory 的 PI05 分支：
        # InjectDefaultPrompt -> ResizeImages(224, 224) -> TokenizePrompt -> PadStatesAndActions(32, 10)。
        model_transforms = ModelTransformFactory()(model_config)

        # 3.2 返回 DataConfig：把基础 DataConfig 与 LIBERO 专属 transforms 合并。
        # Data contract: 返回的三层 transform 让离线样本和在线 obs 走同一口径，避免训练/部署格式漂移。
        # dataclasses.replace 表示：先拿 create_base_config 生成的基础 DataConfig，再把 LIBERO 专属 transforms 填进去。
        # 最终 data_loader.transform_dataset 会按 repack -> data_transforms -> Normalize -> model_transforms 的顺序处理样本。
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Data contract: 可用过滤字典限定每个 episode 保留的 timestep 范围；episode id 由 RLDS metadata 中的
    # recording_folderpath 和 file_path 组成，过滤值是要保留的 (start, end) timestep 区间。

    # Data contract: RLDS 采样源由 name、version、weight 和可选 filter_dict_path 定义。
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Data contract: RLDS DROID 与 LeRobot DROID 原始键名不同，但都要对齐到 DroidInputs 契约：
        # 外部相机、wrist 相机、关节状态、gripper 和 actions。这里的 repack 只服务训练数据加载。
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data contract: joint position action 是绝对目标，训练前需要转成相对当前 state 的 delta。
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Data contract: 自定义 LeRobot DROID 数据应遵守 DroidInputs/DroidOutputs 契约：
        # 7 个机器人维度加 1 个 gripper 维度，并使用与在线 Franka/DROID-style obs 一致的图像 key。
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # Data contract: 这里假设 actions 是 joint velocity，而不是 absolute joint position，
        # 因此不再额外做 DeltaActions；若数据改成 absolute 目标，需要重新启用 delta 转换。
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAgilexDataConfig(DataConfigFactory):
    # Data contract: 为 True 时，送入模型前将 joint 维度转成相对当前 state 的 delta；gripper 保持 absolute。
    use_delta_joint_actions: bool = True
    # Data contract: 输入缺少 "prompt" 时注入该默认 prompt。
    default_prompt: str | None = None

    # Data contract: Agilex repack 把 LeRobot 键名对齐到 AgilexInputs 契约；自定义相机名时，
    # 同步修改这里的 observation.images.* 映射和 AgilexInputs.EXPECTED_CAMERAS。
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.cam_high",
                            "cam_left_wrist": "observation.images.cam_left_wrist",
                            "cam_right_wrist": "observation.images.cam_right_wrist",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
    )
    # Data contract: 从这些 dataset keys 读取 action sequence。
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[agilex_policy.AgilexInputs()],
            outputs=[agilex_policy.AgilexOutputs()],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            use_quantile_norm=False,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotCalvinDataConfig(DataConfigFactory):

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Data contract: CALVIN 将 observation/action 分字段存储，adapter 会把 ee_pos、ee_rot、gripper
        # 拼成统一的 state/action 向量；自定义 CALVIN-like 数据时，优先改这里的 repack key。
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "image": "video.image_base",
                        "wrist_image": "video.image_wrist",
                        "state_ee_pos": "state.ee_pos",
                        "state_ee_rot": "state.ee_rot",
                        "state_gripper": "state.gripper",
                        "action_delta_ee_pos": "action.delta_ee_pos",
                        "action_delta_ee_rot": "action.delta_ee_rot",
                        "action_gripper": "action.gripper",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # Data contract: CALVIN action chunk 从这些 dataset keys 读取并拼接；顺序必须与 CalvinInputs 中的 action 拼接一致。
        action_sequence_keys: Sequence[str] = ("action.delta_ee_pos", "action.delta_ee_rot", "action.gripper")

        data_transforms = _transforms.Group(
            inputs=[calvin_policy.CalvinInputs(model_type=model_config.model_type)],
            outputs=[calvin_policy.CalvinOutputs()],
        )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=action_sequence_keys,
            use_quantile_norm=False,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # config 名称必须唯一，外部通过它引用该配置。
    name: tyro.conf.Suppress[str]
    # 项目名称。
    project_name: str = "openpi"
    # 实验名称，用于命名 metadata 和 checkpoint 目录。
    exp_name: str = tyro.MISSING

    # 定义 model config；action_dim、action_horizon、max_token_len 等共享字段见 BaseModelConfig，
    # 具体模型配置（如 Pi0Config）会继承它并补充额外字段。
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # model 初始化后，可选 weight_loader 从磁盘加载完整或部分权重。
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # 可选 PyTorch checkpoint 路径，用于加载权重。
    pytorch_weight_path: str | None = None

    # PyTorch 训练精度。
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # 指定哪些 weights 需要 freeze。
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # 指定训练使用的数据配置。
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # config assets 的基础目录，例如 norm stats。
    assets_base_dir: str = "./assets"
    # checkpoints 的基础目录。
    checkpoint_base_dir: str = "./checkpoints"

    # 训练中各随机生成器使用的 random seed。
    seed: int = 42
    # 全局 batch size。
    batch_size: int = 32
    # data loader worker 数；增大会加快加载，但也会增加内存和 CPU 占用。
    num_workers: int = 2
    # 训练步数，以 batch 为单位。
    num_train_steps: int = 30_000

    # 每隔多少 step 记录一次训练指标。
    log_interval: int = 100
    # 每隔多少 step 保存一次 checkpoint。
    save_interval: int = 1000
    # 若设置，满足 step % keep_period == 0 的已有 checkpoint 不会被删除。
    keep_period: int | None = 5000

    # 为 True 时，如果 checkpoint 目录已存在则覆盖。
    overwrite: bool = False
    # 为 True 时，从最后一个 checkpoint 继续训练。
    resume: bool = False

    # 为 True 时启用 wandb logging。
    wandb_enabled: bool = True

    # 传给 policy server 的 metadata。
    policy_metadata: dict[str, Any] | None = None

    # 大于 1 时启用 FSDP，并按指定设备数 shard；显存占用会下降，但训练可能变慢。
    # 例如 total device 为 4 且 fsdp_devices 为 2 时，模型 shard 到 2 个设备，并在 2 组设备间做 data parallel。
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# 代码中需要按名称取 config 时使用 `get_config`。
_CONFIGS = [
    #
    # ALOHA 推理配置。
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # DROID 推理配置。
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # LIBERO 微调配置。
    #
    # 这些 train configs 用于在自有 dataset 上微调 base model。
    # 它们定义训练数据、base checkpoint、训练步数、学习率等关键超参数。
    # 自有数据集可复制该配置，并按下方注释调整 dataset 名称和 data transforms。
    TrainConfig(
        # 将 name 改成能反映你的模型和 dataset 的名称。
        name="pi0_libero",
        # 这里定义 model config；本例使用 pi0 架构并执行 full finetuning。
        # 下方示例展示如何改成低显存 LoRA 微调，或改用 pi0-FAST 架构。
        model=pi0_config.Pi0Config(),
        # Data contract: 这里定义训练 dataset；本例使用 LIBERO。
        # 自有 dataset 需要将 repo_id 指向你的数据，并把 DataConfig 换成匹配该 dataset 的配置。
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # Data contract: 为 True 时，从 LeRobot dataset 的 ``task`` 字段加载 prompt，并写入输入 dict 的 ``prompt``。
                # 推荐保持 True。
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # 这里定义用于初始化模型的预训练 checkpoint；应与上面的 model config 匹配。
        # 本例使用 pi0 base model。
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # 其他超参数如 learning rate、训练步数等可在下方定义；完整列表见 TrainConfig。
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # 示例：加载 pi0 model 进行 LoRA fine-tuning。
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # freeze_filter 定义训练时要 freeze 的参数。
        # LoRA 微调时可用 model config 的 helper 生成默认 freeze filter，但必须与上方 model config 匹配。
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # LoRA fine-tuning 关闭 EMA。
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # 示例：加载 pi0-FAST model 进行 full finetuning。
        # Data contract: action_dim 和 action_horizon 需要匹配你的 dataset，action_horizon 即目标 action chunk 长度。
        # max_token_len 是模型可处理的最大非图像 token 数，包括 tokenized prompt、proprioceptive state 和 FAST action tokens。
        # 设得太小会截断序列末尾并触发 warning；设得太大会因为 batch padding 浪费显存。
        # 经验上单臂约 180、双臂约 250，建议先偏小设置，若训练中出现大量 warning 再增大。
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # 这里加载 pi0-FAST base model checkpoint。
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # 示例：加载 pi0-FAST model 进行 LoRA finetuning。
        # action_dim、action_horizon 和 max_token_len 的设置参见上方说明。
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # 提取 LoRA fine-tuning 的 freeze_filter 时，仍需与上方 model config 保持一致。
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # LoRA fine-tuning 关闭 EMA。
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
        num_workers=8,
    ),
    TrainConfig(
        name="pi05_libero_low_mem_finetune",
        # 4090 / 24GB 友好的 LIBERO LoRA 微调配置：保留 pi0.5 数据与模型格式，但只训练 LoRA 参数。
        # 相比 pi05_libero full fine-tuning，这里冻结大部分 base 参数、关闭 EMA，并降低 batch_size。
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            # LoRA 配置和 pi05_libero 使用同一份 LIBERO 数据与同一套 state/action 归一化统计量。
            # 复用 assets/pi05_libero 可以避免为新 config 重新运行 compute_norm_stats。
            assets=AssetsConfig(assets_dir="./assets/pi05_libero", asset_id="physical-intelligence/libero"),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        num_workers=8,
        # freeze_filter 必须由同一份 LoRA model config 生成，确保只训练 LoRA 参数。
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        # LoRA 微调关闭 EMA，避免额外维护一整份参数滑动平均导致显存/内存增加。
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero_lora_from_libero",
        # 从已经在 LIBERO 上微调好的 pi05_libero checkpoint 继续做 LoRA 微调。
        # 与 pi05_libero_low_mem_finetune 的区别只在初始化权重：这里用 pi05_libero，而不是 pi05_base。
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_libero/assets",
                asset_id="physical-intelligence/libero",
            ),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_libero/params"),
        num_train_steps=30_000,
        num_workers=8,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero_diffusion",
        # Diffusion objective version of pi05_libero: same PaliGemma + Gemma action expert architecture,
        # but Pi0Diffusion.compute_loss/sample_actions use q-sample + DDIM instead of flow matching Euler updates.
        model=pi0_config.Pi0DiffusionConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            diffusion_prediction_type="epsilon",
            diffusion_schedule="cosine",
            num_diffusion_train_timesteps=100,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        num_workers=8,
    ),
    TrainConfig(
        name="pi05_libero_diffusion_lora",
        # 24GB 级别显存优先用这个配置做方案三 smoke test；它复用 pi05_libero 的 norm stats。
        model=pi0_config.Pi0DiffusionConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            diffusion_prediction_type="epsilon",
            diffusion_schedule="cosine",
            num_diffusion_train_timesteps=100,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            assets=AssetsConfig(assets_dir="./assets/pi05_libero", asset_id="physical-intelligence/libero"),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        num_workers=8,
        freeze_filter=pi0_config.Pi0DiffusionConfig(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_faster_libero",
        # FASTER: LIBERO 的 FASTER 冒烟测试配置。
        # NOTE: max_delay=0 会关闭历史 action_prefix 采样；若训练遇到 randint 边界问题，优先检查这里。
        model=pi0_config.Pi0FasterConfig(
            pi05=True, action_horizon=10, discrete_state_input=False, max_delay=0, mix_prob=0.5, alpha=0.6, u0=0.9
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        num_workers=8,
    ),
    TrainConfig(
        name="pi05_calvin",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotCalvinDataConfig(
            repo_id="InternRobotics/InternData-Calvin_ABC",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        num_workers=8,
    ),
    TrainConfig(
        name="pi05_faster_calvin",
        # FASTER: CALVIN 复用同一套 FASTER 机制，用来验证更长的语言条件任务链。
        # NOTE: max_delay=0 会默认关闭 prefix mode；若需要测 delay/action_prefix，需要显式改配置。
        model=pi0_config.Pi0FasterConfig(
            pi05=True, action_horizon=10, discrete_state_input=False, max_delay=0, mix_prob=0.5, alpha=0.6, u0=0.9
        ),
        data=LeRobotCalvinDataConfig(
            repo_id="InternRobotics/InternData-Calvin_ABC",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        num_workers=8,
    ),
    #
    # ALOHA 微调配置。
    #
    # 这是自定义 LeRobot dataset 训练流程的测试配置。
    # 自有 ALOHA dataset 的转换和训练说明见 examples/aloha_real/README.md。
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # DROID 微调配置。
    #
    TrainConfig(
        # Data contract: 该配置用于在完整 DROID dataset 上微调 pi0-FAST-base。
        # 使用 RLDS data loading 是为了让大规模 dataset 训练可承载；自有 DROID dataset 微调见下方配置。
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Data contract: 这里应设置为 DROID RLDS dataset 路径，即 `droid` 目录的父目录。
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps 通常足够，在 8x H100 上约需 2 天
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # NOTE: RLDS DataLoader 内部处理 multiprocessing，要求 num_workers=0
    ),
    TrainConfig(
        # Data contract: 该配置用于在完整 DROID dataset 上微调 pi05。
        # 使用 RLDS data loading 是为了让大规模 dataset 训练可承载；自有 DROID dataset 微调见下方配置。
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Data contract: 这里应设置为 DROID RLDS dataset 路径，即 `droid` 目录的父目录。
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # NOTE: RLDS DataLoader 内部处理 multiprocessing，要求 num_workers=0
    ),
    TrainConfig(
        # Data contract: 这个配置用于在自定义小规模 DROID 数据上微调 pi05-DROID，数据格式仍走 LeRobot。
        # 转换脚本见 examples/droid/convert_droid_data_to_lerobot.py。
        name="pi05_droid_finetune",
        # NOTE: 这里是普通 Pi0Config。若要在 Franka/DROID-style 数据上测 FASTER streaming，
        # 应新增 Pi0FasterConfig 配置，而不是直接对它开启 --streaming。
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 使用 32-D actions 训练
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Data contract: 替换成你的自定义 DROID LeRobot dataset repo id。
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # NOTE: 微调 DROID 契约数据时必须复用原始 DROID norm stats，避免 action/state 标度漂移。
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim 配置，用于演示如何在简单仿真环境上训练。
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_agilex",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAgilexDataConfig(
            repo_id="your/dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="asset"),
        ),
        batch_size=128,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=50_000,
        save_interval=10_000,
    ),
    TrainConfig(
        name="pi05_rtc_agilex",
        # FASTER: RTC-style 训练使用 delay/action_prefix 条件，但 mix_prob=0 表示只用 constant schedule，
        # 不混入 HAS。
        model=pi0_config.Pi0FasterConfig(pi05=True, max_delay=10, mix_prob=0.0),
        data=LeRobotAgilexDataConfig(
            repo_id="your/dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="asset"),
        ),
        batch_size=128,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=50_000,
        save_interval=10_000,
    ),
    TrainConfig(
        name="pi05_faster_agilex",
        # FASTER: Agilex 这里启用 max_delay，可同时覆盖 action_prefix 与 HAS 的训练/推理路径。
        model=pi0_config.Pi0FasterConfig(pi05=True, max_delay=10, mix_prob=0.5, alpha=0.6, u0=0.9),
        data=LeRobotAgilexDataConfig(
            repo_id="your/dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="asset"),
        ),
        batch_size=128,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=50_000,
        save_interval=10_000,
    ),
    #
    # 调试配置。
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    # RoboArena 与 PolaRiS 配置。
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
