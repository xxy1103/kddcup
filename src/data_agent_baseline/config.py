from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# 项目根目录用于解析默认数据路径和相对配置路径。
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# 默认公开数据集目录。
def _default_dataset_root() -> Path:
    return PROJECT_ROOT / "data" / "public" / "input"


# 默认运行产物目录。
def _default_run_output_dir() -> Path:
    return PROJECT_ROOT / "artifacts" / "runs"


# 数据集相关配置。
@dataclass(frozen=True, slots=True)
class DatasetConfig:
    root_path: Path = field(default_factory=_default_dataset_root)


# Agent 相关配置，包括模型信息和单任务最大模型轮数。
@dataclass(frozen=True, slots=True)
class AgentConfig:
    model: str = "gpt-4.1-mini"
    api_base: str = "https://api.openai.com/v1"
    api_key: str = ""
    api_key_env: str | None = None
    max_steps: int = 16
    temperature: float = 0.0
    enable_thinking: bool = False
    enable_data_inspector: bool = False


@dataclass(frozen=True, slots=True)
class DataInspectorSampleBudget:
    catalog_sample_rows: int = 5
    max_doc_chars: int = 2000
    max_json_chars: int = 4000


@dataclass(frozen=True, slots=True)
class DataInspectorConfig:
    mode: str = "hybrid"
    inject_summary_to_agent: bool = True
    max_agent_steps: int = 3
    enable_semantic_tools: bool = True
    context_bundle_limit: int = 5
    max_join_hops: int = 3
    sample_budget: DataInspectorSampleBudget = field(default_factory=DataInspectorSampleBudget)


def _optional_string_value(raw_value: object) -> str | None:
    if raw_value is None:
        return None
    normalized = str(raw_value).strip()
    return normalized or None


def _bool_value(raw_value: object, default_value: bool) -> bool:
    if raw_value is None:
        return default_value
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"Expected a boolean value, got: {raw_value!r}")


def _float_value(raw_value: object, default_value: float) -> float:
    if raw_value is None:
        return default_value
    return float(raw_value)


def _string_list_value(raw_value: object, *, field_name: str) -> tuple[str, ...] | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, list):
        raise ValueError(f"Expected `{field_name}` to be a YAML list.")

    normalized_values: list[str] = []
    for raw_item in raw_value:
        item = str(raw_item).strip()
        if item:
            normalized_values.append(item)

    if not normalized_values:
        return None

    # 保序去重，避免重复任务被重复执行。
    return tuple(dict.fromkeys(normalized_values))


def _dotenv_value(dotenv_path: Path, env_var_name: str) -> str | None:
    if not dotenv_path.exists():
        return None

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue

        name, raw_value = line.split("=", 1)
        if name.strip() != env_var_name:
            continue

        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        return value.strip()

    return None


def _resolve_api_key(raw_api_key: object, raw_api_key_env: object, default_api_key: str) -> tuple[str, str | None]:
    api_key = str(raw_api_key if raw_api_key is not None else default_api_key).strip()
    api_key_env = _optional_string_value(raw_api_key_env)
    if api_key or api_key_env is None:
        return api_key, api_key_env

    dotenv_api_key = _dotenv_value(PROJECT_ROOT / ".env", api_key_env)
    return (dotenv_api_key or "").strip(), api_key_env


# 运行时配置，包括输出目录、并发度、任务超时和提交态软时限。
@dataclass(frozen=True, slots=True)
class RunConfig:
    output_dir: Path = field(default_factory=_default_run_output_dir)
    run_id: str | None = None
    max_workers: int = 4
    task_timeout_seconds: int = 600
    soft_runtime_limit_seconds: int = 42300
    task_ids: tuple[str, ...] | None = None


# 顶层应用配置，把 dataset / agent / run 三组配置聚合在一起。
@dataclass(frozen=True, slots=True)
class AppConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    data_inspector: DataInspectorConfig = field(default_factory=DataInspectorConfig)
    run: RunConfig = field(default_factory=RunConfig)


# 提交模式配置：沿用 AppConfig 复用运行逻辑，并额外挂载日志目录。
@dataclass(frozen=True, slots=True)
class SubmissionConfig:
    app_config: AppConfig
    log_dir: Path
    parameter_config_path: Path | None = None

    @property
    def input_root(self) -> Path:
        return self.app_config.dataset.root_path

    @property
    def output_dir(self) -> Path:
        return self.app_config.run.output_dir


# 把 YAML 中的路径字段解析成 Path；相对路径默认相对于项目根目录。
def _path_value(raw_value: str | None, default_value: Path) -> Path:
    if not raw_value:
        return default_value
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def _submission_path_value(env_var_name: str, default_value: str) -> Path:
    raw_value = os.environ.get(env_var_name, default_value).strip()
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    return candidate.resolve()


def _required_env_value(env_var_name: str) -> str:
    value = os.environ.get(env_var_name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {env_var_name}")
    return value


def resolve_submission_log_dir_from_env() -> Path:
    return _submission_path_value("DABENCH_LOG_ROOT", "/logs")


def resolve_submission_parameter_config_path_from_env() -> Path | None:
    explicit_path = _optional_string_value(os.environ.get("DABENCH_SUBMISSION_CONFIG"))
    if explicit_path is not None:
        resolved_path = _path_value(explicit_path, PROJECT_ROOT / "configs" / "submission.yaml")
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Submission parameter config does not exist: {resolved_path}")
        return resolved_path

    default_path = PROJECT_ROOT / "configs" / "submission.yaml"
    if default_path.is_file():
        return default_path
    return None


def _load_submission_parameter_payload(config_path: Path | None) -> dict[str, object]:
    if config_path is None:
        return {}

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Submission parameter config must contain a YAML object at the top level.")

    allowed_top_level_keys = {"agent", "data_inspector", "run"}
    unexpected_top_level_keys = set(payload) - allowed_top_level_keys
    if unexpected_top_level_keys:
        raise ValueError(
            "Submission parameter config only supports top-level keys "
            f"{sorted(allowed_top_level_keys)}, got {sorted(unexpected_top_level_keys)}."
        )

    agent_payload = payload.get("agent", {})
    data_inspector_payload = payload.get("data_inspector", {})
    run_payload = payload.get("run", {})
    if not isinstance(agent_payload, dict) or not isinstance(data_inspector_payload, dict) or not isinstance(run_payload, dict):
        raise ValueError(
            "Submission parameter config sections `agent`, `data_inspector`, and `run` must be YAML objects."
        )

    allowed_agent_keys = {"max_steps", "temperature", "enable_thinking", "enable_data_inspector"}
    allowed_data_inspector_keys = {
        "mode",
        "inject_summary_to_agent",
        "max_agent_steps",
        "enable_semantic_tools",
        "context_bundle_limit",
        "max_join_hops",
        "sample_budget",
    }
    allowed_run_keys = {"max_workers", "task_timeout_seconds", "soft_runtime_limit_seconds"}
    unexpected_agent_keys = set(agent_payload) - allowed_agent_keys
    unexpected_data_inspector_keys = set(data_inspector_payload) - allowed_data_inspector_keys
    unexpected_run_keys = set(run_payload) - allowed_run_keys
    if unexpected_agent_keys:
        raise ValueError(
            "Submission parameter config only supports non-sensitive `agent` keys "
            f"{sorted(allowed_agent_keys)}, got {sorted(unexpected_agent_keys)}."
        )
    if unexpected_data_inspector_keys:
        raise ValueError(
            "Submission parameter config only supports non-sensitive `data_inspector` keys "
            f"{sorted(allowed_data_inspector_keys)}, got {sorted(unexpected_data_inspector_keys)}."
        )
    if unexpected_run_keys:
        raise ValueError(
            "Submission parameter config only supports non-sensitive `run` keys "
            f"{sorted(allowed_run_keys)}, got {sorted(unexpected_run_keys)}."
        )

    return payload


def _data_inspector_mode_value(raw_value: object, default_value: str) -> str:
    mode = str(raw_value if raw_value is not None else default_value).strip().lower()
    if mode not in {"rules", "hybrid"}:
        raise ValueError("data_inspector.mode must be either `rules` or `hybrid`.")
    return mode


def _data_inspector_sample_budget_value(raw_value: object | None) -> DataInspectorSampleBudget:
    defaults = DataInspectorSampleBudget()
    if raw_value is None:
        return defaults
    if not isinstance(raw_value, dict):
        raise ValueError("data_inspector.sample_budget must be a YAML object.")
    return DataInspectorSampleBudget(
        catalog_sample_rows=int(raw_value.get("catalog_sample_rows", defaults.catalog_sample_rows)),
        max_doc_chars=int(raw_value.get("max_doc_chars", defaults.max_doc_chars)),
        max_json_chars=int(raw_value.get("max_json_chars", defaults.max_json_chars)),
    )


def _data_inspector_config_value(raw_value: object | None) -> DataInspectorConfig:
    defaults = DataInspectorConfig()
    if raw_value is None:
        return defaults
    if not isinstance(raw_value, dict):
        raise ValueError("data_inspector must be a YAML object.")
    return DataInspectorConfig(
        mode=_data_inspector_mode_value(raw_value.get("mode"), defaults.mode),
        inject_summary_to_agent=_bool_value(
            raw_value.get("inject_summary_to_agent"),
            defaults.inject_summary_to_agent,
        ),
        max_agent_steps=int(raw_value.get("max_agent_steps", defaults.max_agent_steps)),
        enable_semantic_tools=_bool_value(
            raw_value.get("enable_semantic_tools"),
            defaults.enable_semantic_tools,
        ),
        context_bundle_limit=int(raw_value.get("context_bundle_limit", defaults.context_bundle_limit)),
        max_join_hops=int(raw_value.get("max_join_hops", defaults.max_join_hops)),
        sample_budget=_data_inspector_sample_budget_value(raw_value.get("sample_budget")),
    )


def load_submission_config_from_env() -> SubmissionConfig:
    agent_defaults = AgentConfig()
    run_defaults = RunConfig()

    input_root = _submission_path_value("DABENCH_INPUT_ROOT", "/input")
    output_root = _submission_path_value("DABENCH_OUTPUT_ROOT", "/output")
    log_dir = resolve_submission_log_dir_from_env()
    parameter_config_path = resolve_submission_parameter_config_path_from_env()
    parameter_payload = _load_submission_parameter_payload(parameter_config_path)
    agent_parameter_payload = parameter_payload.get("agent", {})
    data_inspector_parameter_payload = parameter_payload.get("data_inspector", {})
    run_parameter_payload = parameter_payload.get("run", {})

    if not input_root.is_dir():
        raise FileNotFoundError(f"Submission input root does not exist: {input_root}")

    agent_config = AgentConfig(
        model=_required_env_value("MODEL_NAME"),
        api_base=_required_env_value("MODEL_API_URL"),
        api_key=_required_env_value("MODEL_API_KEY"),
        api_key_env=None,
        max_steps=int(os.environ.get("DABENCH_MAX_STEPS", agent_parameter_payload.get("max_steps", agent_defaults.max_steps))),
        temperature=_float_value(
            os.environ.get("DABENCH_TEMPERATURE", agent_parameter_payload.get("temperature")),
            agent_defaults.temperature,
        ),
        enable_thinking=_bool_value(
            os.environ.get("DABENCH_ENABLE_THINKING", agent_parameter_payload.get("enable_thinking")),
            agent_defaults.enable_thinking,
        ),
        enable_data_inspector=_bool_value(
            os.environ.get(
                "DABENCH_ENABLE_DATA_INSPECTOR",
                agent_parameter_payload.get("enable_data_inspector"),
            ),
            agent_defaults.enable_data_inspector,
        ),
    )
    data_inspector_config = _data_inspector_config_value(data_inspector_parameter_payload)
    run_config = RunConfig(
        output_dir=output_root,
        run_id=None,
        max_workers=int(os.environ.get("DABENCH_MAX_WORKERS", run_parameter_payload.get("max_workers", run_defaults.max_workers))),
        task_timeout_seconds=int(
            os.environ.get(
                "DABENCH_TASK_TIMEOUT_SECONDS",
                run_parameter_payload.get("task_timeout_seconds", run_defaults.task_timeout_seconds),
            )
        ),
        soft_runtime_limit_seconds=int(
            os.environ.get(
                "DABENCH_SOFT_RUNTIME_LIMIT_SECONDS",
                run_parameter_payload.get("soft_runtime_limit_seconds", run_defaults.soft_runtime_limit_seconds),
            )
        ),
    )
    return SubmissionConfig(
        app_config=AppConfig(
            dataset=DatasetConfig(root_path=input_root),
            agent=agent_config,
            data_inspector=data_inspector_config,
            run=run_config,
        ),
        log_dir=log_dir,
        parameter_config_path=parameter_config_path,
    )


# 从 YAML 配置文件加载应用配置，并对缺省值和相对路径做统一处理。
def load_app_config(config_path: Path) -> AppConfig:
    payload = yaml.safe_load(config_path.read_text()) or {}
    dataset_defaults = DatasetConfig()
    agent_defaults = AgentConfig()
    run_defaults = RunConfig()

    dataset_payload = payload.get("dataset", {})
    agent_payload = payload.get("agent", {})
    data_inspector_payload = payload.get("data_inspector", {})
    run_payload = payload.get("run", {})

    dataset_config = DatasetConfig(
        root_path=_path_value(dataset_payload.get("root_path"), dataset_defaults.root_path),
    )
    api_key, api_key_env = _resolve_api_key(
        agent_payload.get("api_key", agent_defaults.api_key),
        agent_payload.get("api_key_env"),
        agent_defaults.api_key,
    )
    agent_config = AgentConfig(
        model=str(agent_payload.get("model", agent_defaults.model)),
        api_base=str(agent_payload.get("api_base", agent_defaults.api_base)),
        api_key=api_key,
        api_key_env=api_key_env,
        max_steps=int(agent_payload.get("max_steps", agent_defaults.max_steps)),
        temperature=_float_value(agent_payload.get("temperature"), agent_defaults.temperature),
        enable_thinking=_bool_value(agent_payload.get("enable_thinking"), agent_defaults.enable_thinking),
        enable_data_inspector=_bool_value(
            agent_payload.get("enable_data_inspector"),
            agent_defaults.enable_data_inspector,
        ),
    )
    data_inspector_config = _data_inspector_config_value(data_inspector_payload)
    raw_run_id = run_payload.get("run_id")
    run_id = run_defaults.run_id
    if raw_run_id is not None:
        normalized_run_id = str(raw_run_id).strip()
        # 空字符串视为未指定 run_id，交由 runner 自动生成。
        run_id = normalized_run_id or None

    run_config = RunConfig(
        output_dir=_path_value(run_payload.get("output_dir"), run_defaults.output_dir),
        run_id=run_id,
        max_workers=int(run_payload.get("max_workers", run_defaults.max_workers)),
        task_timeout_seconds=int(run_payload.get("task_timeout_seconds", run_defaults.task_timeout_seconds)),
        soft_runtime_limit_seconds=int(
            run_payload.get("soft_runtime_limit_seconds", run_defaults.soft_runtime_limit_seconds)
        ),
        task_ids=_string_list_value(run_payload.get("task_ids"), field_name="run.task_ids"),
    )
    return AppConfig(
        dataset=dataset_config,
        agent=agent_config,
        data_inspector=data_inspector_config,
        run=run_config,
    )
