import argparse
import json
import logging
from pathlib import Path
import shutil
from typing import List, Optional

from pydicom import dcmread
from pydicom.dataset import Dataset, FileDataset
from pydicom.errors import InvalidDicomError

from radifox.records.hashing import hash_file_dir

from ._version import __version__
from .anondb import AnonDB
from .deanon import deanonymize_subject
from .exec import run_conversion, ExecError
from .lut import LookupTable
from .metadata import Metadata
from .utils import silentremove, mkdir_p, version_check


def _extract_patient_info(ds: Dataset | FileDataset) -> dict:
    """Extract patient-level DICOM attributes from a pydicom Dataset."""
    return {
        "patient_id": getattr(ds, "PatientID", None),
        "patient_name": str(getattr(ds, "PatientName", "")) or None,
        "patient_birth_date": getattr(ds, "PatientBirthDate", None),
        "patient_sex": getattr(ds, "PatientSex", None),
        "study_uid": getattr(ds, "StudyInstanceUID", None),
        "institution_name": getattr(ds, "InstitutionName", None),
    }


def _find_first_dicom(directory: Path):
    """Walk a directory to find and read the first valid DICOM file."""
    for f in sorted(directory.rglob("*")):
        if f.is_file():
            try:
                ds = dcmread(f, stop_before_pixels=True)
                return ds
            except (InvalidDicomError, Exception):
                continue
    return None


def _run_deanonymize(args: argparse.Namespace) -> None:
    """Reverse anonymization using the mapping database."""
    project_id = args.project_id.upper()
    project_dir = args.output_root / project_id.lower()

    if not project_dir.exists():
        raise ValueError("Project directory does not exist: %s" % project_dir)

    with AnonDB(args.anon_db) as db:
        subjects = db.get_all_subjects()
        if args.subject:
            subjects = [s for s in subjects if s.patient_id == args.subject]
            if not subjects:
                raise ValueError("Patient ID '%s' not found in database." % args.subject)

        for subject in subjects:
            sessions = db.get_sessions_for_subject(subject.anon_id)
            deanonymize_subject(project_dir, project_id, subject, sessions)

    print("\n--- De-anonymized %d subject(s) ---" % len(subjects))


def convert(args: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, nargs="?", help="Source directory/file to convert.")
    parser.add_argument(
        "-o", "--output-root", type=Path, help="Output root directory.", required=True
    )
    parser.add_argument("-l", "--lut-file", type=Path, help="Lookup table file.")
    parser.add_argument("-p", "--project-id", type=str, help="Project ID.")
    parser.add_argument("-s", "--subject-id", type=str, help="Subject ID.")
    parser.add_argument("-e", "--session-id", type=str, help="Session ID.")
    parser.add_argument("--site-id", type=str, help="Site ID.")
    parser.add_argument("--tms-metafile", type=Path, help="TMS metadata file.")
    parser.add_argument("--verbose", action="store_true", help="Verbose output.")
    parser.add_argument(
        "--force", action="store_true", help="Force run even if it would be skipped."
    )
    parser.add_argument(
        "--reckless", action="store_true", help="Force run and overwrite existing data."
    )
    parser.add_argument(
        "--safe", action="store_true", help="Add -N to session ID, if session exists."
    )
    parser.add_argument(
        "--no-project-subdir", action="store_true", help="Do not create project subdirectory."
    )
    parser.add_argument("--parrec", action="store_true", help="Source is PARREC.")
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create symbolic links to source data instead of copying.",
    )
    parser.add_argument(
        "--hardlink",
        action="store_true",
        help="Create hard links to source data instead of copying.",
    )
    parser.add_argument("--institution", type=str, help="Institution name.")
    parser.add_argument("--field-strength", type=int, help="Magnetic field strength.")
    parser.add_argument(
        "--force-dicom", action="store_true", help="Force read DICOM files.", default=False
    )
    parser.add_argument(
        "--force-derived",
        action="store_true",
        help="Convert derived/secondary DICOM series that would normally be skipped.",
        default=False,
    )
    parser.add_argument(
        "--anonymize",
        action="store_true",
        help="Anonymize DICOM data (irreversible unless --anon-db is also provided).",
    )
    parser.add_argument("--date-shift-days", type=int, help="Number of days to shift dates.")
    parser.add_argument(
        "--anon-db",
        type=Path,
        help="Path to SQLite anonymization mapping database. "
        "Records original patient identifiers for later de-anonymization. "
        "Requires --anonymize.",
    )
    parser.add_argument(
        "--deanonymize",
        action="store_true",
        help="Reverse anonymization using the mapping database "
        "(requires --anon-db and --project-id).",
    )
    parser.add_argument(
        "--subject",
        type=str,
        help="De-anonymize only this patient ID (only with --deanonymize).",
    )
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)

    args = parser.parse_args(args)

    for argname in ["source", "output_root", "lut_file", "tms_metafile", "anon_db"]:
        if getattr(args, argname, None) is not None:
            setattr(args, argname, getattr(args, argname).resolve())

    # --- Deanonymize mode ---
    if args.deanonymize:
        if args.anon_db is None:
            raise ValueError("--deanonymize requires --anon-db.")
        if args.project_id is None:
            raise ValueError("--deanonymize requires --project-id.")
        _run_deanonymize(args)
        return

    if args.subject:
        raise ValueError("--subject is only valid with --deanonymize.")

    if args.source is None:
        raise ValueError("source is required (omit only with --deanonymize).")

    if args.anon_db is not None and not args.anonymize:
        raise ValueError("--anon-db requires --anonymize.")

    if args.date_shift_days is not None and not args.anonymize:
        raise ValueError("--date-shift-days requires --anonymize.")

    # --- Standard conversion ---
    if args.hardlink and args.symlink:
        raise ValueError("Only one of --symlink and --hardlink can be used.")
    linking = "hardlink" if args.hardlink else ("symlink" if args.symlink else None)

    mapping = {"subject_id": "SubjectID", "session_id": "SessionID", "site_id": "SiteID"}
    if args.tms_metafile:
        metadata = Metadata.from_tms_metadata(args.tms_metafile, args.no_project_subdir)
        for argname in ["subject_id", "session_id", "site_id"]:
            if getattr(args, argname) is not None:
                setattr(metadata, mapping[argname], getattr(args, argname))
    else:
        for argname in ["project_id", "subject_id", "session_id"]:
            if getattr(args, argname) is None:
                raise ValueError(
                    "%s is a required argument when no metadata file is provided." % argname
                )
        metadata = Metadata(
            args.project_id,
            args.subject_id,
            args.session_id,
            args.site_id,
            args.no_project_subdir,
        )

    if args.lut_file is None:
        lut_file = (
            (args.output_root / (metadata.projectname + "-lut.csv"))
            if args.no_project_subdir
            else (args.output_root / metadata.projectname / (metadata.projectname + "-lut.csv"))
        )
    else:
        lut_file = args.lut_file

    manual_json_file = (
        args.output_root / metadata.dir_to_str() / (metadata.prefix_to_str() + "_ManualNaming.json")
    )
    manual_names = json.loads(manual_json_file.read_text()) if manual_json_file.exists() else {}

    type_dirname = "%s" % "parrec" if args.parrec else "dcm"
    if (args.output_root / metadata.dir_to_str() / type_dirname).exists():
        if args.safe:
            metadata.AttemptNum = 2
            while (args.output_root / metadata.dir_to_str() / type_dirname).exists():
                metadata.AttemptNum += 1
        elif args.force or args.reckless:
            if not args.reckless:
                json_file = (
                    args.output_root
                    / metadata.dir_to_str()
                    / (metadata.prefix_to_str() + "_UnconvertedInfo.json")
                )
                if not json_file.exists():
                    raise ValueError(
                        "Unconverted info file (%s) does not exist for consistency checking. "
                        "Cannot use --force, use --reckless instead." % json_file
                    )
                json_obj = json.loads(json_file.read_text())
                if json_obj["Metadata"]["TMSMetaFileHash"] is not None:
                    if metadata.TMSMetaFileHash is None:
                        raise ValueError(
                            "Previous conversion did not use a TMS metadata file, "
                            "run with --reckless to ignore this error."
                        )
                    if json_obj["Metadata"]["TMSMetaFileHash"] != metadata.TMSMetaFileHash:
                        raise ValueError(
                            "TMS meta data file has changed since last conversion, "
                            "run with --reckless to ignore this error."
                        )
                elif (
                    json_obj["Metadata"]["TMSMetaFileHash"] is None
                    and metadata.TMSMetaFileHash is not None
                ):
                    raise ValueError(
                        "Previous conversion used a TMS metadata file, "
                        "run with --reckless to ignore this error."
                    )
                if hash_file_dir(args.source, False) != json_obj["InputHash"]:
                    raise ValueError(
                        "Source file(s) have changed since last conversion, "
                        "run with --reckless to ignore this error."
                    )
            shutil.rmtree(args.output_root / metadata.dir_to_str() / type_dirname)
            silentremove(args.output_root / metadata.dir_to_str() / "nii")
            for filepath in (args.output_root / metadata.dir_to_str()).glob("*.json"):
                silentremove(filepath)
        else:
            raise RuntimeError(
                "Output directory exists, run with --force to remove outputs and re-run."
            )

    manual_arg = {
        "MagneticFieldStrength": args.field_strength,
        "InstitutionName": args.institution,
    }

    # If --anon-db is provided, read DICOM patient info before conversion
    # (source files may be removed during anonymized conversion)
    if args.anon_db is not None:
        ds = _find_first_dicom(args.source) if args.source.is_dir() else None
        patient_info = _extract_patient_info(ds) if ds is not None else {}

    run_conversion(
        args.source,
        args.output_root,
        metadata,
        lut_file,
        args.verbose,
        args.parrec,
        False,
        linking,
        manual_arg,
        args.force_dicom,
        args.anonymize,
        args.date_shift_days,
        manual_names,
        None,
        args.force_derived,
    )

    # Record mapping in anonymization database after successful conversion
    if args.anon_db is not None:
        with AnonDB(args.anon_db) as db:
            anon_id = db.get_or_create_subject(
                patient_id=patient_info.get("patient_id") or metadata.SubjectID,
                patient_name=patient_info.get("patient_name"),
                patient_birth_date=patient_info.get("patient_birth_date"),
                patient_sex=patient_info.get("patient_sex"),
                date_shift_days=args.date_shift_days,
                anon_id=metadata.SubjectID,
            )
            db.add_session(
                anon_id=anon_id,
                source_path=str(args.source),
                original_study_uid=patient_info.get("study_uid"),
                institution_name=patient_info.get("institution_name"),
            )
            db.commit()
        print("Recorded mapping in %s (subject %s)" % (args.anon_db, anon_id))


def update(args: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path, help="Existing RADIFOX Directory to update.")
    parser.add_argument("-l", "--lut-file", type=Path, help="Lookup table file.")
    parser.add_argument(
        "--force", action="store_true", help="Force run even if it would be skipped."
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output.")
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)

    args = parser.parse_args(args)

    session_id = args.directory.name
    subj_id = args.directory.parent.name

    json_file = args.directory / "_".join([subj_id, session_id, "UnconvertedInfo.json"])
    if not json_file.exists():
        safe_json_file = args.directory / "_".join(
            [subj_id, "-".join(session_id.split("-")[:-1]), "UnconvertedInfo.json"]
        )
        if not safe_json_file.exists():
            raise ValueError("Unconverted info file (%s) does not exist." % json_file)
        json_file = safe_json_file
    json_obj = json.loads(json_file.read_text())

    metadata = Metadata.from_dict(json_obj["Metadata"])
    if session_id != metadata.SessionID:
        metadata.AttemptNum = int(session_id.split("-")[-1])
    # noinspection PyProtectedMember
    output_root = (
        Path(*args.directory.parts[:-2])
        if metadata._NoProjectSubdir
        else Path(*args.directory.parts[:-3])
    )

    if args.lut_file is None:
        # noinspection PyProtectedMember
        if metadata._NoProjectSubdir:
            lut_file = output_root / (metadata.projectname + "-lut.csv")
        else:
            lut_file = output_root / metadata.projectname / (metadata.projectname + "-lut.csv")
    else:
        lut_file = args.lut_file
    lookup_dict = (
        LookupTable(lut_file, metadata.ProjectID, metadata.SiteID).LookupDict
        if lut_file.exists()
        else {}
    )

    manual_json_file = args.directory / (metadata.prefix_to_str() + "_ManualNaming.json")
    manual_names = json.loads(manual_json_file.read_text()) if manual_json_file.exists() else {}

    if not args.force and (
        version_check(json_obj["__version__"]["radifox"], __version__)
        and json_obj["LookupTable"]["LookupDict"] == lookup_dict
        and json_obj["ManualNames"] == manual_names
    ):
        print(
            "No action required. Software version, LUT dictionary and naming dictionary match for %s."
            % args.directory
        )
        return

    parrec = (args.directory / "parrec").exists()
    type_dir = args.directory / ("%s" % "parrec" if parrec else "dcm")

    mkdir_p(args.directory / "prev")
    for filename in ["nii", "qa", json_file.name]:
        if (args.directory / filename).exists():
            (args.directory / filename).rename(args.directory / "prev" / filename)
    try:
        run_conversion(
            type_dir,
            output_root,
            metadata,
            lut_file,
            args.verbose,
            parrec,
            True,
            None,
            json_obj.get("ManualArgs", {}),
            False,
            False,
            0,
            manual_names,
            json_obj["InputHash"],
        )
    except ExecError:
        logging.info("Exception caught during update. Resetting to previous state.")
        for filename in ["nii", "qa", json_file.name]:
            silentremove(args.directory / filename)
            if (args.directory / "prev" / filename).exists():
                (args.directory / "prev" / filename).rename(args.directory / filename)
    else:
        for dirname in ["stage", "proc"]:
            if (args.directory / dirname).exists():
                (args.directory / dirname / "CHECK").touch()
    silentremove(args.directory / "prev")


# TODO: Add "rename" command to rename sessions
