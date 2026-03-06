import json
from pathlib import Path

from .anondb import Session, Subject
from .utils import shift_date


def _patch_json_sidecar(json_file: Path, patient_id: str, session: Session, subject: Subject) -> bool:
    """Patch a single JSON sidecar file to restore original patient identifiers.

    Returns True if the file was changed.
    """
    json_obj = json.loads(json_file.read_text())
    changed = False

    if "Metadata" in json_obj:
        json_obj["Metadata"]["SubjectID"] = patient_id
        json_obj["RemoveIdentifiers"] = False
        changed = True

    if "SeriesList" in json_obj:
        for series in json_obj["SeriesList"]:
            if session.source_path and series.get("SourcePath") is None:
                series["SourcePath"] = session.source_path
                changed = True
            if session.institution_name is not None:
                series["InstitutionName"] = session.institution_name
                changed = True
            if session.original_study_uid is not None:
                series["StudyUID"] = session.original_study_uid
                changed = True
            if subject.date_shift_days and series.get("AcqDateTime"):
                series["AcqDateTime"] = shift_date(series["AcqDateTime"], -subject.date_shift_days)
                changed = True

    if changed:
        json_file.write_text(json.dumps(json_obj, indent=4, sort_keys=True))
        print(f"  Patched {json_file.name}")

    return changed


def _rename_files(directory: Path, old_prefix: str, new_prefix: str, label: str = "") -> None:
    """Rename all files in a directory that start with old_prefix to use new_prefix."""
    if old_prefix == new_prefix:
        return
    prefix = (f"  {label}") if label else " "
    for filepath in sorted(directory.iterdir()):
        if filepath.name.startswith(old_prefix):
            new_name = filepath.name.replace(old_prefix, new_prefix, 1)
            new_path = directory / new_name
            if new_path.exists():
                print(f" {prefix}Skipping {filepath.name}: target {new_name} already exists")
                continue
            filepath.rename(new_path)
            print(f" {prefix}Renamed {filepath.name} -> {new_name}")


def deanonymize_subject(project_dir: Path, project_id: str, subject: Subject, sessions: list[Session]) -> None:
    """De-anonymize a single subject: patch JSONs, rename files, rename directory."""
    anon_id_upper = subject.anon_id.upper()
    patient_id_upper = subject.patient_id.upper()
    anon_dir_name = f"{project_id}-{anon_id_upper}"
    new_dir_name = f"{project_id}-{patient_id_upper}"
    anon_subject_dir = project_dir / anon_dir_name

    if not anon_subject_dir.exists():
        print(f"Skipping {anon_dir_name}: directory not found.")
        return

    print(f"{anon_dir_name} -> {new_dir_name}")

    for sess in sessions:
        session_dir = anon_subject_dir / sess.session_id
        if not session_dir.exists():
            continue

        anon_prefix = f"{project_id}-{anon_id_upper}_{sess.session_id}"
        new_prefix = f"{project_id}-{patient_id_upper}_{sess.session_id}"

        for json_file in session_dir.glob("*.json"):
            _patch_json_sidecar(json_file, patient_id_upper, sess, subject)

        _rename_files(session_dir, anon_prefix, new_prefix)

        nii_dir = session_dir / "nii"
        if nii_dir.exists():
            _rename_files(nii_dir, anon_prefix, new_prefix, label="nii/")

        qa_dir = session_dir / "qa"
        if qa_dir.exists():
            for qa_subdir in sorted(qa_dir.iterdir()):
                if qa_subdir.is_dir():
                    _rename_files(qa_subdir, anon_prefix, new_prefix, label="qa/%s/" % qa_subdir.name)

    if anon_dir_name != new_dir_name:
        new_subject_dir = project_dir / new_dir_name
        anon_subject_dir.rename(new_subject_dir)
        print(f"  Renamed directory {anon_dir_name} -> {new_dir_name}")
