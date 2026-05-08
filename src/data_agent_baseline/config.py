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
    model_env: str | None = None
    api_base: str = "https://api.openai.com/v1"
    api_base_env: str | None = None
    api_key: str = ""
    api_key_env: str | None = None
    model_request_timeout_seconds: float | None = 120.0
    max_steps: int = 16
    temperature: float = 0.0
    enable_data_inspector: bool = False


@dataclass(frozen=True, slots=True)
class DataInspectorSampleBudget:
    catalog_sample_rows: int = 5
    catalog_top_distinct_values: int = 50
    max_doc_chars: int = 2000
    max_json_chars: int = 4000


@dataclass(frozen=True, slots=True)
class DataInspectorConfig:
    mode: str = "hybrid"
    inject_summary_to_agent: bool = True
    max_agent_steps: int = 5
    max_phase_retries: int = 1
    profile_guided_fast_path: bool = True
    enable_semantic_tools: bool = True
    include_inspector_trace: bool = True
    context_bundle_limit: int = 6
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


def _optional_timeout_value(raw_value: object, default_value: float | None) -> float | None:
    if raw_value is None:
        return default_value
    if isinstance(raw_value, str) and raw_value.strip().lower() in {"", "none", "null"}:
        return None
    value = float(raw_value)
    if value <= 0:
        return None
    return value


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


def _env_value(env_var_name: str) -> str | None:
    value = os.environ.get(env_var_name, "").strip()
    return value or None


def _resolve_env_backed_string(
    *,
    raw_value: object,
    raw_env: object,
    default_value: str,
    field_name: str,
) -> tuple[str, str | None]:
    env_var_name = _optional_string_value(raw_env)
    if env_var_name is not None:
        value = _env_value(env_var_name)
        if value is None:
            raise ValueError(f"Missing environment variable `{env_var_name}` for `{field_name}`.")
        return value, env_var_name
    return str(raw_value if raw_value is not None else default_value).strip(), None


def _resolve_api_key(raw_api_key: object, raw_api_key_env: object, default_api_key: str) -> tuple[str, str | None]:
    api_key = str(raw_api_key if raw_api_key is not None else default_api_key).strip()
    api_key_env = _optional_string_value(raw_api_key_env)
    if api_key or api_key_env is None:
        return api_key, api_key_env

    env_api_key = _env_value(api_key_env)
    if env_api_key is not None:
        return env_api_key, api_key_env

    dotenv_api_key = _dotenv_value(PROJECT_ROOT / ".env", api_key_env)
    return (dotenv_api_key or "").strip(), api_key_env


# 运行时配置，包括输出目录、并发度、任务超时和提交态软时限。
@dataclass(frozen=True, slots=True)
class RunConfig:
    output_dir: Path = field(default_factory=_default_run_output_dir)
    log_dir: Path | None = None
    output_layout: str = "run_dir"
    run_id: str | None = None
    max_workers: int = 4
    task_timeout_seconds: int = 600
    task_ids: tuple[str, ...] | None = None


# 顶层应用配置，把 dataset / agent / run 三组配置聚合在一起。
@dataclass(frozen=True, slots=True)
class AppConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    data_inspector: DataInspectorConfig = field(default_factory=DataInspectorConfig)
    run: RunConfig = field(default_factory=RunConfig)


# 把 YAML 中的路径字段解析成 Path；相对路径默认相对于项目根目录。
def _path_value(raw_value: str | None, default_value: Path) -> Path:
    if not raw_value:
        return default_value
    raw_text = str(raw_value).strip()
    if not raw_text:
        return default_value
    candidate = Path(raw_text)
    if candidate.is_absolute() or raw_text.startswith("/"):
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def _output_layout_value(raw_value: object, default_value: str) -> str:
    output_layout = str(raw_value if raw_value is not None else default_value).strip().lower()
    if output_layout not in {"run_dir", "flat"}:
        raise ValueError("run.output_layout must be either `run_dir` or `flat`.")
    return output_layout


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
        catalog_top_distinct_values=int(raw_value.get("catalog_top_distinct_values", defaults.catalog_top_distinct_values)),
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
        max_phase_retries=int(raw_value.get("max_phase_retries", defaults.max_phase_retries)),
        profile_guided_fast_path=_bool_value(
            raw_value.get("profile_guided_fast_path"),
            defaults.profile_guided_fast_path,
        ),
        enable_semantic_tools=_bool_value(
            raw_value.get("enable_semantic_tools"),
            defaults.enable_semantic_tools,
        ),
        include_inspector_trace=_bool_value(
            raw_value.get("include_inspector_trace"),
            defaults.include_inspector_trace,
        ),
        context_bundle_limit=int(raw_value.get("context_bundle_limit", defaults.context_bundle_limit)),
        sample_budget=_data_inspector_sample_budget_value(raw_value.get("sample_budget")),
    )


# 从 YAML 配置文件加载应用配置，并对缺省值和相对路径做统一处理。
def load_app_config(config_path: Path) -> AppConfig:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
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
    model, model_env = _resolve_env_backed_string(
        raw_value=agent_payload.get("model", agent_defaults.model),
        raw_env=agent_payload.get("model_env"),
        default_value=agent_defaults.model,
        field_name="agent.model",
    )
    api_base, api_base_env = _resolve_env_backed_string(
        raw_value=agent_payload.get("api_base", agent_defaults.api_base),
        raw_env=agent_payload.get("api_base_env"),
        default_value=agent_defaults.api_base,
        field_name="agent.api_base",
    )
    api_key, api_key_env = _resolve_api_key(
        agent_payload.get("api_key", agent_defaults.api_key),
        agent_payload.get("api_key_env"),
        agent_defaults.api_key,
    )
    agent_config = AgentConfig(
        model=model,
        model_env=model_env,
        api_base=api_base,
        api_base_env=api_base_env,
        api_key=api_key,
        api_key_env=api_key_env,
        model_request_timeout_seconds=_optional_timeout_value(
            agent_payload.get("model_request_timeout_seconds"),
            agent_defaults.model_request_timeout_seconds,
        ),
        max_steps=int(agent_payload.get("max_steps", agent_defaults.max_steps)),
        temperature=_float_value(agent_payload.get("temperature"), agent_defaults.temperature),
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
        log_dir=(
            _path_value(run_payload.get("log_dir"), run_defaults.output_dir)
            if run_payload.get("log_dir") is not None
            else None
        ),
        output_layout=_output_layout_value(run_payload.get("output_layout"), run_defaults.output_layout),
        run_id=run_id,
        max_workers=int(run_payload.get("max_workers", run_defaults.max_workers)),
        task_timeout_seconds=int(run_payload.get("task_timeout_seconds", run_defaults.task_timeout_seconds)),
        task_ids=_string_list_value(run_payload.get("task_ids"), field_name="run.task_ids"),
    )
    return AppConfig(
        dataset=dataset_config,
        agent=agent_config,
        data_inspector=data_inspector_config,
        run=run_config,
    )
