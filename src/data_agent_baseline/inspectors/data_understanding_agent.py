from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.config import DataInspectorConfig
from data_agent_baseline.inspectors.exchange import (
    AgentEnvelope,
    AgentEnvelopeContent,
    ArtifactRef,
    ResourceRef,
    SemanticClaim,
    Uncertainty,
)
from data_agent_baseline.inspectors.handoff import (
    AnswerContract,
    DataFabric,
    DataUnderstandingHandoff,
    GroundedConcept,
    GroundedField,
    HandoffUncertainty,
    JoinEdge,
    JoinPath,
    QuestionGrounding,
    RejectedField,
    SemanticNotesDraft,
)
from data_agent_baseline.inspectors.prompts import (
    SEMANTIC_SYNTHESIS_SYSTEM_PROMPT,
    build_semantic_synthesis_retry_prompt,
    build_semantic_synthesis_prompt,
)
from data_agent_baseline.inspectors.semantic_catalog import build_semantic_catalog
from data_agent_baseline.inspectors.semantic_index import build_semantic_index
from data_agent_baseline.inspectors.semantic_query import SemanticQueryTools


@dataclass(frozen=True, slots=True)
class DataUnderstandingResult:
    perception: dict[str, Any]
    semantic_catalog: dict[str, Any]
    semantic_index: dict[str, Any]
    perception_envelope: dict[str, Any]
    semantic_context_envelope: dict[str, Any]
    data_understanding_handoff: dict[str, Any]
    summary: str
    synthesis: dict[str, Any] | None
    synthesis_error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "perception": self.perception,
            "semantic_catalog": self.semantic_catalog,
            "semantic_index": self.semantic_index,
            "perception_envelope": self.perception_envelope,
            "semantic_context_envelope": self.semantic_context_envelope,
            "data_understanding_handoff": self.data_understanding_handoff,
            "summary": self.summary,
            "synthesis": self.synthesis,
            "synthesis_error": self.synthesis_error,
        }


class DataUnderstandingAgent:
    def __init__(self, *, model: Any | None, config: DataInspectorConfig) -> None:
        self.model = model
        self.config = config

    def run(self, task: PublicTask, perception_envelope: AgentEnvelope) -> DataUnderstandingResult:
        perception_payload = dict(perception_envelope.content.payload)
        catalog = build_semantic_catalog(task, budget=self.config.sample_budget)
        semantic_index = build_semantic_index(
            question=task.question,
            catalog=catalog,
            perception_payload=perception_payload,
        )
        query_tools = SemanticQueryTools(
            catalog=catalog,
            semantic_index=semantic_index,
            limit=self.config.context_bundle_limit,
            max_join_hops=self.config.max_join_hops,
        )
        context_bundle = (
            query_tools.build_context_bundle(task.question)
            if self.config.enable_semantic_tools
            else {
                "query": task.question,
                "field_candidates": catalog.get("query_relevance", {}).get("relevant_fields", []),
                "join_paths": catalog.get("relationships", []),
                "knowledge_hits": [],
                "rejected_field_hints": [],
                "risk_candidates": semantic_index.get("risk_index", {}),
            }
        )
        notes, notes_error = self._maybe_synthesize(
            task=task,
            perception_payload=perception_payload,
            context_bundle=context_bundle,
            field_whitelist=sorted(query_tools.known_field_refs()) if self.config.enable_semantic_tools else [],
        )
        handoff = _build_handoff(
            task=task,
            perception_payload=perception_payload,
            context_bundle=context_bundle,
            notes=notes,
            notes_error=notes_error,
        )
        envelope = self._build_semantic_context_envelope(
            task=task,
            catalog=catalog,
            semantic_index=semantic_index,
            perception_envelope=perception_envelope,
            handoff=handoff,
        )
        return DataUnderstandingResult(
            perception=perception_payload,
            semantic_catalog=catalog,
            semantic_index=semantic_index,
            perception_envelope=perception_envelope.to_dict(),
            semantic_context_envelope=envelope.to_dict(),
            data_understanding_handoff=handoff.to_dict(),
            summary=envelope.content.summary,
            synthesis=None if notes is None else notes.model_dump(mode="json"),
            synthesis_error=notes_error,
        )

    def _maybe_synthesize(
        self,
        *,
        task: PublicTask,
        perception_payload: dict[str, Any],
        context_bundle: dict[str, Any],
        field_whitelist: list[str],
    ) -> tuple[SemanticNotesDraft | None, str | None]:
        if self.config.mode != "hybrid" or self.config.max_agent_steps <= 0 or self.model is None:
            return None, None
        prompt = build_semantic_synthesis_prompt(
            question=task.question,
            perception_payload=perception_payload,
            context_bundle=context_bundle,
            field_whitelist=field_whitelist,
        )
        errors: list[str] = []
        try:
            return _invoke_and_validate_notes(
                model=self.model,
                prompt=prompt,
                field_whitelist=field_whitelist,
            )
        except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
            errors.append(str(exc))
            retry_prompt = build_semantic_synthesis_retry_prompt(
                previous_error=str(exc),
                question=task.question,
                context_bundle=context_bundle,
                field_whitelist=field_whitelist,
            )
            try:
                draft, _ = _invoke_and_validate_notes(
                    model=self.model,
                    prompt=retry_prompt,
                    field_whitelist=field_whitelist,
                )
                return draft, f"Semantic synthesis recovered after retry. First error: {errors[0]}"
            except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as retry_exc:
                errors.append(str(retry_exc))
                return None, f"Semantic synthesis discarded after retry: {' | '.join(errors)}"
            except Exception as retry_exc:  # noqa: BLE001
                errors.append(str(retry_exc))
                return None, f"Semantic synthesis failed after retry: {' | '.join(errors)}"
        except Exception as exc:  # noqa: BLE001
            return None, f"Semantic synthesis failed: {exc}"

    def _build_semantic_context_envelope(
        self,
        *,
        task: PublicTask,
        catalog: dict[str, Any],
        semantic_index: dict[str, Any],
        perception_envelope: AgentEnvelope,
        handoff: DataUnderstandingHandoff,
    ) -> AgentEnvelope:
        resources = [
            ResourceRef(
                id=f"res_{index}",
                kind=str(asset.get("kind", "file")),
                path=str(asset.get("path", "")),
                schema_ref=f"schema:{asset.get('path')}",
                recommended_tools=list(asset.get("recommended_tools", [])),
            )
            for index, asset in enumerate(catalog.get("assets", []))
        ]
        artifacts = [
            ArtifactRef(id="perception", kind="json", path="perception.json"),
            ArtifactRef(id="semantic_catalog", kind="json", path="semantic_catalog.json"),
            ArtifactRef(id="semantic_index", kind="json", path="semantic_index.json"),
            ArtifactRef(id="data_understanding_handoff", kind="json", path="data_understanding_handoff.json"),
        ]
        claims = list(perception_envelope.content.semantic_claims)
        uncertainties = list(perception_envelope.content.uncertainties)
        claims.extend(_claims_from_relationships(catalog))
        uncertainties.extend(_uncertainties_from_catalog(catalog))
        claims.extend(SemanticClaim(claim=note, confidence="low") for note in handoff.llm_notes)
        uncertainties.extend(
            Uncertainty(risk=item.risk, instruction=item.instruction, evidence_refs=item.evidence_refs)
            for item in handoff.uncertainties
        )

        summary = handoff.brief_markdown
        return AgentEnvelope(
            task_id=task.task_id,
            sender="DataUnderstandingAgent",
            recipient="MainSolvingAgent",
            message_type="semantic_context",
            content=AgentEnvelopeContent(
                summary=summary,
                resources=resources,
                artifacts=artifacts,
                semantic_claims=claims,
                uncertainties=uncertainties,
                payload={
                    "catalog_asset_count": len(catalog.get("assets", [])),
                    "catalog_schema_count": len(catalog.get("schemas", [])),
                    "answer_contract": handoff.answer_contract.model_dump(mode="json"),
                    "handoff": handoff.to_dict(),
                },
            ),
        )


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if match is None:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Semantic synthesis must be a JSON object.")
    return payload


def _claims_from_relationships(catalog: dict[str, Any]) -> list[SemanticClaim]:
    claims: list[SemanticClaim] = []
    for relationship in catalog.get("relationships", [])[:10]:
        field = relationship.get("field")
        locations = relationship.get("locations", [])
        claims.append(
            SemanticClaim(
                claim=f"Field `{field}` is a possible join key across {len(locations)} assets.",
                confidence=str(relationship.get("confidence", "medium")),
                evidence_refs=[
                    f"field:{item.get('asset_path')}:{item.get('field')}" for item in locations
                ],
            )
        )
    return claims


def _uncertainties_from_catalog(catalog: dict[str, Any]) -> list[Uncertainty]:
    return [
        Uncertainty(
            risk=str(item.get("risk", "catalog_uncertainty")),
            instruction=str(item.get("instruction", item)),
            evidence_refs=[str(item.get("asset_path"))] if item.get("asset_path") else [],
        )
        for item in catalog.get("semantic_uncertainties", [])
    ]


def _invoke_and_validate_notes(
    *,
    model: Any,
    prompt: str,
    field_whitelist: list[str],
) -> tuple[SemanticNotesDraft, None]:
    response = model.invoke(
        [
            SystemMessage(content=SEMANTIC_SYNTHESIS_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )
    text = _message_text(getattr(response, "content", response))
    payload = _extract_json_object(text)
    draft = SemanticNotesDraft.model_validate(payload)
    _validate_notes_against_whitelist(draft, field_whitelist)
    return draft, None


def _validate_notes_against_whitelist(draft: SemanticNotesDraft, field_whitelist: list[str]) -> None:
    if not field_whitelist:
        return
    text = "\n".join([*draft.semantic_notes, *draft.uncertainties])
    references = re.findall(r"\b(?:csv|json|db|doc)/[A-Za-z0-9_./-]+(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b", text)
    asset_whitelist = {_asset_ref_from_field_ref(ref) for ref in field_whitelist}
    unknown = [ref for ref in references if ref not in field_whitelist and ref not in asset_whitelist]
    if unknown:
        raise ValueError(f"LLM referenced fields outside whitelist: {unknown[:5]}")


def _asset_ref_from_field_ref(field_ref: str) -> str:
    for extension in (".csv", ".json", ".db", ".md", ".txt"):
        marker = field_ref.find(extension)
        if marker >= 0:
            return field_ref[: marker + len(extension)]
    return field_ref.rsplit(".", 1)[0]


def _build_handoff(
    *,
    task: PublicTask,
    perception_payload: dict[str, Any],
    context_bundle: dict[str, Any],
    notes: SemanticNotesDraft | None,
    notes_error: str | None,
) -> DataUnderstandingHandoff:
    concepts = _build_grounded_concepts(task.question, context_bundle)
    join_paths = _build_join_paths(context_bundle)
    answer_contract = _build_answer_contract(task.question, concepts)
    uncertainties = _build_handoff_uncertainties(perception_payload, context_bundle, notes)
    llm_notes = [] if notes is None else list(notes.semantic_notes)
    handoff = DataUnderstandingHandoff(
        brief_markdown="",
        question_grounding=QuestionGrounding(concepts=concepts),
        data_fabric=DataFabric(join_paths=join_paths),
        answer_contract=answer_contract,
        uncertainties=uncertainties,
        llm_notes=llm_notes,
        llm_notes_error=notes_error,
    )
    return handoff.model_copy(update={"brief_markdown": _render_handoff_brief(handoff)})


def _build_grounded_concepts(question: str, context_bundle: dict[str, Any]) -> list[GroundedConcept]:
    question_tokens = set(_tokenize(question))
    field_candidates = context_bundle.get("field_candidates", [])
    rejected_hints = context_bundle.get("rejected_field_hints", [])
    concepts: list[GroundedConcept] = []

    if "event" in question_tokens:
        concepts.append(
            GroundedConcept(
                term="event",
                role="answer_entity",
                grounded_fields=_grounded_fields_for_terms(field_candidates, ["event_name", "event_id", "event"]),
            )
        )
    if "cost" in question_tokens:
        concepts.append(
            GroundedConcept(
                term="cost",
                role="metric",
                grounded_fields=_grounded_fields_for_terms(field_candidates, ["cost"]),
                rejected_fields=[
                    RejectedField(
                        asset=str(item.get("asset", "")),
                        field=str(item.get("field", "")),
                        reason=str(item.get("reason", "")),
                    )
                    for item in rejected_hints
                ],
            )
        )
    if question_tokens & {"lowest", "minimum", "min"}:
        concepts.append(
            GroundedConcept(
                term="lowest",
                role="operation",
                grounded_fields=[],
            )
        )
    if question_tokens & {"highest", "maximum", "max"}:
        concepts.append(
            GroundedConcept(
                term="highest",
                role="operation",
                grounded_fields=[],
            )
        )
    return concepts


def _grounded_fields_for_terms(field_candidates: list[dict[str, Any]], terms: list[str]) -> list[GroundedField]:
    grounded: list[GroundedField] = []
    normalized_terms = {_normalize_text(term) for term in terms}
    for item in sorted(field_candidates, key=lambda value: _field_priority(str(value.get("field", "")), normalized_terms)):
        field = str(item.get("field", ""))
        normalized_field = _normalize_text(field)
        if not any(term in normalized_field for term in normalized_terms):
            continue
        grounded_item = GroundedField(
            asset=str(item.get("asset_path", "")),
            field=field,
            confidence="high" if _normalize_text(field).endswith(tuple(normalized_terms)) else "medium",
            reason="Matched question term to catalog/index field candidate.",
        )
        if grounded_item not in grounded:
            grounded.append(grounded_item)
    return grounded[:6]


def _field_priority(field: str, normalized_terms: set[str]) -> int:
    normalized_field = _normalize_text(field)
    if normalized_field.endswith("eventname"):
        return 0
    if normalized_field.endswith("cost"):
        return 0
    if any(normalized_field.endswith(term) for term in normalized_terms):
        return 1
    return 2


def _build_join_paths(context_bundle: dict[str, Any]) -> list[JoinPath]:
    paths: list[JoinPath] = []
    for path_payload in context_bundle.get("join_paths", []):
        edges = [
            JoinEdge(
                from_field=str(edge.get("from", "")),
                to_field=str(edge.get("to", "")),
                confidence=str(edge.get("confidence", "medium")),
                reason=str(edge.get("reason", "")),
            )
            for edge in path_payload.get("path", [])
        ]
        if not edges:
            continue
        paths.append(
            JoinPath(
                source=str(path_payload.get("source", "")),
                target=str(path_payload.get("target", "")),
                path=edges,
                confidence=str(path_payload.get("confidence", "medium")),
            )
        )
    return paths[:5]


def _build_answer_contract(question: str, concepts: list[GroundedConcept]) -> AnswerContract:
    tokens = set(_tokenize(question))
    metric_operation = "unknown"
    row_policy = "unknown"
    if tokens & {"lowest", "minimum", "min"}:
        metric_operation = "min"
        row_policy = "preserve_all_ties"
    elif tokens & {"highest", "maximum", "max"}:
        metric_operation = "max"
        row_policy = "preserve_all_ties"
    elif question.lower().startswith(("what ", "what's ", "which ")):
        row_policy = "single"

    columns: list[str] = []
    if "event" in tokens:
        columns = ["event_name"]

    metric_field = None
    for concept in concepts:
        if concept.role != "metric":
            continue
        for field in concept.grounded_fields:
            metric_field = f"{field.asset}.{field.field}"
            break
    return AnswerContract(
        columns=columns,
        row_policy=row_policy,
        metric_operation=metric_operation,
        metric_field=metric_field,
    )


def _build_handoff_uncertainties(
    perception_payload: dict[str, Any],
    context_bundle: dict[str, Any],
    notes: SemanticNotesDraft | None,
) -> list[HandoffUncertainty]:
    uncertainties = [
        HandoffUncertainty(
            risk=str(risk),
            instruction=f"Verify high-risk term before final answer: {risk}",
        )
        for risk in perception_payload.get("high_risk_terms", [])
    ]
    if not context_bundle.get("join_paths"):
        uncertainties.append(
            HandoffUncertainty(
                risk="missing_join_path",
                instruction="No deterministic join path was found; verify relationships manually before answering.",
            )
        )
    if notes is not None:
        uncertainties.extend(
            HandoffUncertainty(risk="llm_semantic_uncertainty", instruction=item)
            for item in notes.uncertainties
        )
    return uncertainties


def _render_handoff_brief(handoff: DataUnderstandingHandoff) -> str:
    lines = ["## Data Understanding Brief", ""]
    if handoff.answer_contract.columns:
        lines.append(f"Question asks for: {', '.join(handoff.answer_contract.columns)}")
    if handoff.answer_contract.metric_field:
        lines.append(f"Metric field: {handoff.answer_contract.metric_field}")
    if handoff.answer_contract.metric_operation != "unknown":
        lines.append(
            f"Metric operation: {handoff.answer_contract.metric_operation}; row policy: {handoff.answer_contract.row_policy}"
        )

    lines.extend(["", "Grounding:"])
    for concept in handoff.question_grounding.concepts:
        if concept.grounded_fields:
            rendered_fields = ", ".join(f"{field.asset}.{field.field}" for field in concept.grounded_fields[:4])
            lines.append(f"- {concept.term} -> {rendered_fields}")
        for rejected in concept.rejected_fields[:4]:
            lines.append(f"- Do not use {rejected.asset}.{rejected.field}: {rejected.reason}")

    if handoff.data_fabric.join_paths:
        lines.extend(["", "Join path:"])
        for join_path in handoff.data_fabric.join_paths[:3]:
            rendered_edges = " -> ".join(f"{edge.from_field} => {edge.to_field}" for edge in join_path.path)
            lines.append(f"- {rendered_edges}")

    if handoff.answer_contract.columns:
        lines.extend(["", "Answer contract:"])
        lines.append(f"- Submit only columns: {', '.join(handoff.answer_contract.columns)}")
        if handoff.answer_contract.row_policy == "preserve_all_ties":
            lines.append("- Preserve all rows tied at the extreme value.")
    if handoff.llm_notes:
        lines.extend(["", "Optional LLM notes:"])
        lines.extend(f"- {note}" for note in handoff.llm_notes[:5])
    lines.append("")
    lines.append("Verify this handoff with tools before final answer.")
    return "\n".join(lines)[:4000]


def _tokenize(text: str) -> list[str]:
    return [token for token in re.split(r"[^A-Za-z0-9]+", text.lower()) if token]


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())
