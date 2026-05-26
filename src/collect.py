from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from api_client import LectureApiClient, LectureApiError
from config import CHECKPOINT_PATH, RAW_DATA_DIR, SUBJECTS_PATH


# subjects.csv의 lectureId 열에서 고유 강의 코드를 한 건씩 읽어옵니다.
def iter_lecture_ids(path: Path = SUBJECTS_PATH) -> Iterator[int]:
    """Yield unique lectureId values while reading one CSV row at a time."""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = csv.reader(file)

        try:
            header = next(rows)
        except StopIteration:
            raise ValueError(f"Expected a header row in {path}, but the file is empty.") from None

        try:
            lecture_id_index = header.index("lectureId")
        except ValueError:
            raise ValueError(f"Expected a 'lectureId' column in {path}.") from None

        seen_ids: set[int] = set()

        for row_number, row in enumerate(rows, start=2):
            if lecture_id_index >= len(row):
                raise ValueError(f"Expected a lectureId value in {path} at row {row_number}.")

            lecture_id = row[lecture_id_index].strip()
            if not lecture_id:
                continue

            try:
                parsed_id = int(lecture_id)
            except ValueError as exc:
                raise ValueError(
                    f"Expected an integer lectureId in {path} at row {row_number}, "
                    f"but got {lecture_id!r}."
                ) from exc

            if parsed_id not in seen_ids:
                seen_ids.add(parsed_id)
                yield parsed_id


# API 응답 데이터를 보기 좋은 JSON 파일로 저장합니다.
def save_json(data: dict[str, Any], path: Path) -> None:
    """Save one API response as a pretty-printed JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


# 기존 강의평가 파일이 있으면 JSON 객체로 읽어옵니다.
def load_saved_json(path: Path) -> dict[str, Any] | None:
    """Load an existing saved API response when the output file exists."""
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}.")

    return data


# 강의평가 응답에서 검증된 article 목록을 꺼냅니다.
def get_articles(response: dict[str, Any], source: str) -> list[Any]:
    """Return an article list from a validated API response object."""
    result = response.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("articles"), list):
        raise ValueError(f"Expected {source} article response to contain a result.articles list.")

    return result["articles"]


# 저장에 불필요한 사용자별 article 메타데이터를 제거합니다.
def sanitize_article(article: dict[str, Any]) -> dict[str, Any]:
    """Remove per-user fields before saving an article."""
    return {
        key: value
        for key, value in article.items()
        if key != "isMine"
    }


# 두 강의평가 응답을 id 기준으로 합치고 저장 대상 필드만 남깁니다.
def merge_article_responses(
    existing: dict[str, Any] | None,
    incoming: dict[str, Any],
) -> dict[str, Any]:
    """Merge articles by id, then remove fields that are not useful in output."""
    incoming_articles = get_articles(incoming, "incoming")
    existing_articles = [] if existing is None else get_articles(existing, "existing")

    merged_articles: list[dict[str, Any]] = []
    seen_article_ids: set[Any] = set()

    for article in [*existing_articles, *incoming_articles]:
        if not isinstance(article, dict):
            raise ValueError("Expected each article item to be a JSON object.")

        article_id = article.get("id")
        if article_id is not None and article_id in seen_article_ids:
            continue

        if article_id is not None:
            seen_article_ids.add(article_id)
        elif sanitize_article(article) in merged_articles:
            continue

        merged_articles.append(sanitize_article(article))

    merged = dict(incoming)
    merged_result = dict(incoming.get("result", {}))
    merged_result["articles"] = merged_articles
    merged["result"] = merged_result
    return merged


# 새 강의평가 응답을 기존 파일의 article 목록과 병합해 저장합니다.
def save_merged_articles(data: dict[str, Any], path: Path) -> None:
    """Save article data after merging it with already collected reviews."""
    existing = load_saved_json(path)
    merged = merge_article_responses(existing, data)
    save_json(merged, path)


# 현재 완료된 수집 상태를 다음 재개용 체크포인트로 저장합니다.
def save_checkpoint(
    completed_lecture_ids: list[int],
    completed_request_count: int,
    batch_size: int,
    path: Path = CHECKPOINT_PATH,
) -> None:
    """Atomically save collection progress at a completed lecture boundary."""
    checkpoint = {
        "version": 1,
        "batch_size": batch_size,
        "completed_request_count": completed_request_count,
        "completed_lecture_ids": completed_lecture_ids,
    }
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    save_json(checkpoint, temporary_path)
    temporary_path.replace(path)


# 저장된 체크포인트에서 완료 강의와 요청 진행 상태를 복원합니다.
def load_checkpoint(path: Path = CHECKPOINT_PATH) -> tuple[list[int], int]:
    """Load completed lecture ids and request count for a resumed collection."""
    checkpoint = load_saved_json(path)
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint exists at {path}.")

    lecture_ids = checkpoint.get("completed_lecture_ids")
    request_count = checkpoint.get("completed_request_count")
    if not isinstance(lecture_ids, list) or not all(
        isinstance(lecture_id, int) for lecture_id in lecture_ids
    ):
        raise ValueError(f"Expected integer completed_lecture_ids in {path}.")
    if not isinstance(request_count, int) or request_count < 0:
        raise ValueError(f"Expected a non-negative completed_request_count in {path}.")

    return lecture_ids, request_count


# 완료된 lecture id는 건너뛰고 아직 필요한 대상만 전달합니다.
def iter_pending_lecture_ids(
    lecture_ids: Iterable[int],
    completed_lecture_ids: set[int],
) -> Iterator[int]:
    """Yield lecture ids that are not already recorded as completed."""
    for lecture_id in lecture_ids:
        if lecture_id not in completed_lecture_ids:
            yield lecture_id


# lecture id별 상세 데이터와 글 목록 데이터를 수집합니다.
def collect(
    lecture_ids: Iterable[int] | None = None,
    resume: bool = False,
    checkpoint_path: Path = CHECKPOINT_PATH,
    client: LectureApiClient | None = None,
) -> None:
    """Collect detail and article-list JSON files for each lecture id."""
    api_client = LectureApiClient() if client is None else client
    all_source_ids = iter_lecture_ids() if lecture_ids is None else lecture_ids

    if resume:
        completed_lecture_ids, completed_request_count = load_checkpoint(checkpoint_path)
        api_client.completed_request_count = completed_request_count
        print(
            f"[RESUME] completed_lectures={len(completed_lecture_ids)}, "
            f"completed_requests={completed_request_count}"
        )
    else:
        completed_lecture_ids = []
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    completed_id_set = set(completed_lecture_ids)
    source_ids = iter_pending_lecture_ids(all_source_ids, completed_id_set)
    checkpoint_batch = (
        api_client.completed_request_count // api_client.batch_size
        if api_client.batch_size > 0
        else 0
    )

    for lecture_id in source_ids:
        try:
            bundle = api_client.fetch_lecture_bundle(lecture_id)
        except LectureApiError as exc:
            print(f"[FAIL] lecture_id={lecture_id}: {exc}")
        else:
            detail_path = RAW_DATA_DIR / f"lecture_{lecture_id}_detail.json"
            articles_path = RAW_DATA_DIR / f"lecture_{lecture_id}_articles.json"

            save_json(bundle["detail"], detail_path)
            save_merged_articles(bundle["articles"], articles_path)
            completed_lecture_ids.append(lecture_id)
            completed_id_set.add(lecture_id)

            print(f"[OK] lecture_id={lecture_id} -> {detail_path}")
            print(f"[OK] lecture_id={lecture_id} -> {articles_path}")

        current_batch = (
            api_client.completed_request_count // api_client.batch_size
            if api_client.batch_size > 0
            else 0
        )
        if current_batch > checkpoint_batch:
            save_checkpoint(
                completed_lecture_ids,
                api_client.completed_request_count,
                api_client.batch_size,
                checkpoint_path,
            )
            checkpoint_batch = current_batch
            print(f"[CHECKPOINT] completed_lectures={len(completed_lecture_ids)}")

    save_checkpoint(
        completed_lecture_ids,
        api_client.completed_request_count,
        api_client.batch_size,
        checkpoint_path,
    )
    print(f"[DONE] completed_lectures={len(completed_lecture_ids)}")


# 명령행 옵션을 읽어 수집 실행 설정을 구성합니다.
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for collection execution."""
    parser = argparse.ArgumentParser(description="Collect lecture review API data.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue after lecture ids already saved in the checkpoint file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    collect(resume=arguments.resume)
