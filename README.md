# RADIFOX Conversion Tools
The RADIFOX conversion tools are a set of command line scripts and Python modules for converting DICOM (and PARREC) files to NIfTI files using the RADIFOX naming system.
These tools are a wrapper around the `dcm2niix` tool that use the RADIFOX naming system to organize the output files.
The conversion tools also include automatic logging and QA image generation.
JSON sidecar files are created for each NIfTI file that contain critical metadata and conversion information.

## Table of Contents
- [Installation](#installation)
- [Basic Usage](#basic-usage)
  - [CLI Scripts](#cli-scripts)
    - [`radifox-convert`](#radifox-convert)
    - [`radifox-update`](#radifox-update)
- [Conversion](#conversion)
  - [Look-up Tables](#look-up-tables)
  - [Manual Naming](#manual-naming)
  - [JSON Sidecar Files](#json-sidecar-files)
- [Additional Information](#additional-information)
    - [Advanced CLI Usage](#advanced-cli-usage)
        - [`radifox-convert`](#radifox-convert)
        - [`radifox-update`](#radifox-update)
    - [JSON Sidecar Format](#json-sidecar-format)

## Installation
`radifox-convert` is available on PyPI and can be installed with pip:
```bash
pip install radifox-convert
```

## Basic Usage
### CLI Scripts
The `radifox-convert` package includes a number of CLI scripts to convert data and update conversions.
These scripts are installed to your PATH when you install the `radifox-convert` package.
For a full listing of command line options, see [Advanced CLI Usage](#advanced-cli-usage).

#### `radifox-convert`
The `radifox-convert` script is used to convert DICOM files to NIfTI files using the `dcm2niix` tool.
It is a wrapper around `dcm2niix` that uses the RADIFOX naming system to organize the output files.

Example Usage:
```bash
radifox-convert \
    --output-root /path/to/output \
    --project-id study \
    --subject-id 123456 \
    --session-id 1 \
    /path/to/dicom_files
```
This will copy the files in the direction `/path/to/dicom_files` to the output directory `/path/to/output/study/123456/STUDY-1/dcm`, organize them and convert them to NIfTI.
The NIfTI files (and their JSON sidecar files) will be placed in `/path/to/output/study/STUDY-123456/1/nii`.

#### `radifox-update`
The `radifox-update` script is used to update naming for a directory of images.
This is commonly done after an update to RADIFOX to ensure that all images are named according to the latest version of the naming system.
It also could be done to incorporate a new look-up table or manual naming entries after QA.

Example Usage:
```bash
radifox-update --directory /path/to/output/study/STUDY-123456/1
```
This will update the naming for all images in the existing RADIFOX session directory `/path/to/output/study/STUDY-123456/1`.
If the RADIFOX version, look-up table, or manual naming entries have changed, the images will be renamed to reflect the new information.
If none of these have changed, the update will be skipped.

# Conversion
The conversion system is a wrapper around the `dcm2niix` tool.
It uses the RADIFOX naming system to organize the output files.
`radifox-convert` is the core command for this function.

The conversion process is as follows:
 1. Copy the DICOM files to the `dcm` directory in the session directory.
 2. Sort the DICOM files into series directories in the `dcm` directory and remove any duplicates.
 3. Check for series that should be skipped (scouts, localizers, derived images, etc.).
 4. Generate image names automatically from the DICOM metadata, look-up tables, and manual naming entries.
 5. Convert the DICOM files to NIfTI using `dcm2niix` and rename to RADIFOX naming.
 6. Create the JSON sidecar files for the NIfTI files (contains some DICOM metadata).
 7. Create QA images for the converted NIfTI files.

## Look-up Tables
The look-up tables are a set of rules for automatically naming images based on the DICOM `SeriesDescription` tag.
They are stored in a comma-separated values (CSV) file in each project folder.
They have a specific name format: `<project-id>_lut.csv`.
If no look-up table is found for a project, a blank look-up table is written.
Look-up table values take precidence over automatic naming, but are overwritten by manual names.

The look-up table file has five total columns: `Project`, `Site`, `InstitutionName`, `SeriesDescription`, and `OutputFilename`.

The first three columns (`Project`, `Site`, and `InstitutionName`) narrow down which images are affected.
These columns match the project and site IDs and the DICOM `InstitutionName` tag.
This means that if a particular site or even scanning center uses a specific `SeriesDescription`, it can be handled differently than others.
The `Site` and `InstitutionName` columns are optional and can be `None`.

The `SeriesDescription` column is a string and must **exactly** match the DICOM `SeriesDescription` tag.
This may mean that multiple rows are needed to cover all possible values of the `SeriesDescription` tag for a particular name.

The `OutputFilename` column is where the RADIFOX naming is specified.
You do not have to specify all components of the name, only the ones that need to be changed.
For example, if you only want to change the `bodypart` to `BRAIN` for a specific `SeriesDescription`, you can specify `BRAIN` in the `OutputFilename` column.
However, you must specify all components that come prior to the one you want to change as `None`.
For example, to change the `modality` to `T1` for a specific `SeriesDescription`, you must specify `None-T1` in the `OutputFilename` column.
This can also be used to change the `extras`, by specifying them at the end of the `OutputFilename` column.
For example, to add `ECHO1` to the end of the name for a specific `SeriesDescription`, but change nothing else, you must specify `None-None-None-None-None-None-ECHO1` in the `OutputFilename` column.

## Manual Naming
Manual naming entries are the most specific way to name images.
They are stored as a JSON file in each session directory (`<subject-id>_<session-id>_ManualNaming.json`).
This JSON file is a dictionary with the DICOM series directory path (`dcm/...`) as the key and the new name as the value.
This series path can be found as the `SourcePath` in the sidecar JSON file for the image.
Manual naming entries take precidence over look-up tables and automatic naming.
The naming convention for manual naming entries is the same as for look-up tables.

The simplest way to create manual naming entries is to use the `radifox-qa` webapp.

## JSON Sidecar Files
JSON sidecar files are created for each NIfTI file during conversion.
They contain information about the conversion process (versions, look-up table values, manual naming, etc.) as well as critical DICOM metadata.
The JSON sidecar files are stored in the `nii` directory in eact session directory next to their corresponding NIfTI file.

Sidecar files are human-readable, but can also be accessed in Python using the `json` standard package.
Most of the crutial information will be in the `SeriesInfo` key of the sidecar file.

```python
import json

obj = json.load(open('/path/to/output/study/STUDY-123456/1/nii/STUDY-123456_01-03_BRAIN-T1-IRFSPGR-3D-SAGITTAL-PRE.json'))
print(obj['SeriesInfo']['SeriesDescription']) # prints 'IRFSPGR 3D SAGITTAL PRE'
print(obj('SeriesInfo')['SliceThickness']) # prints 1.0
```

A complete record of the sidecar JSON format is below [JSON Sidecar Format](#json-sidecar-format).

## Automatic Logging
The auto-provenance system also includes automatic logging during execution.
This is done by setting up a `logging` handler that writes to the `logs` directory in the session directory.
This handler is set up by default to log all messages to the `logs/<module-name>/<first-input-filename>-<timestamp>-info.log` file.
This can be adjusted to `logs/<module-name>-<timestamp>-info.log` by setting `log_uses_filename` to `False` in the `ProcessingModule` subclass.
Currently, there is support for `INFO`, `WARNING` and `ERROR` level messages.
They can be accessed at any point in the `run` method by calling `logging.info(message)` (or `warning` or `error`).
You must import `logging` at the top of the file to use this feature.
If there are warnings or errors produced during execution, they will be written to additional log files (`-warning.log` and `-error.log`) for easy viewing.
There is currently no support for `DEBUG` level messages, but that is planned for the future.

## Automatic QA Images
The auto-provenance system also includes automatic generation of QA images from outputs.
Any output that is returned from the `run` method will have a QA image generated automatically, if it is a NIfTI file (ends in `.nii.gz`).

# Additional Information

## Advanced CLI Usage
### `radifox-convert`
| Option                      | Description                                                                                                                            | Default                                           |
|-----------------------------|----------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------|
| `source`                    | The source directory (or zip file) containing the DICOM files.                                                                         | `required`                                        |
| `-o`, `--output-root`       | The root directory for the output files (contains project directories).                                                                | `required`                                        |
| `-p`, `--project-id`        | The project ID for the converted session.                                                                                              | `required`                                        |
| `-s`, `--subject-id`        | The subject ID for the converted session.                                                                                              | `required`                                        |
| `-e`, `--session-id`        | The session ID for the converted session.                                                                                              | `required`                                        |
| `-l`, `--lut-file`          | The look-up table file to use for naming.                                                                                              | `<output-root>/<project-id>/<project-id>_lut.csv` |
| `--site-id`                 | The site ID for the converted session.                                                                                                 | `None`                                            |
| `--force`                   | Force conversion even if session directory already exists.                                                                             | `False`                                           |
| `--reckless`                | Skip consistency checks when forcing run (will overwrite files!)                                                                       | `False`                                           |
| `--safe`                    | If the session directory already exists, use a new directory with `-#` appended (does not change session ID or filenames)              | `False`                                           |
| `--no-project-subdir`       | Do not create a project subdirectory in the output root directory. Subjects will be placed directly into the `--output-root` directory | `False`                                           |
| `--symlink`                 | Create symlinks to the original DICOM files instead of copying them.                                                                   | `False`                                           |
| `--hardlink`                | Create hardlinks to the original DICOM files instead of copying them.                                                                  | `False`                                           |
| `--verbose`                 | Log debug output.                                                                                                                      | `False`                                           |
| `--version`                 | Output RADIFOX version and exit.                                                                                                       | `False`                                           |
| `--help`                    | Show help message and exit.                                                                                                            | `False`                                           |
| `--parrec`                  | Convert PAR/REC files instead of DICOM files.                                                                                          | `False`                                           |
| `--institution-name`        | The institution name to use for the session (required for PAR/REC conversion).                                                         | `None`                                            |
| `--magnetic-field-strength` | The magnetic field strength to use for the session (required for PAR/REC conversion).                                                  | `None`                                            |
| `--anonymize`               | Experimental anonymization support (will remove copied DICOM files).                                                                   | `False`                                           |
| `--date-shift-days`         | The number of days to shift the date by during anonymization.                                                                          | `None`                                            |
| `--tms-metafile`            | The TMS metafile to use for subject, site and session ID.                                                                              | `None`                                            |

### `radifox-update`
| Option             | Description                                                | Default                               |
|--------------------|------------------------------------------------------------|---------------------------------------|
| `directory`        | The converted RADIFOX directory to update.                 | `required`                            |
| `-l`, `--lut-file` | The look-up table file to use for naming.                  | `<directory>/../<project-id>_lut.csv` |
| `--force`          | Force conversion even if session directory already exists. | `False`                               |
| `--verbose`        | Log debug output.                                          | `False`                               |
| `--version`        | Output RADIFOX version and exit.                           | `False`                               |
| `--help`           | Show help message and exit.                                | `False`                               |

## JSON Sidecar Format
The JSON sidecar format is a dictionary with 8 top-level keys:
 - `__version__`: A dictionary of software versions used in conversion (`radifox` and `dcm2niix`)
 - `InputHash`: A hash of the input directory or archive file used in conversion
 - `LookupTable`: A dictionary of look-up table values used in conversion (limited by project/site ID/institution, if applicable)
 - `ManualNames`: A dictionary of manual name entries used in conversion
 - `Metadata`: A dictionary of session level metadata items (Project ID, Subject ID, Session ID, etc.)
 - `RemoveIdentifiers`: A boolean indicating if identifiers were removed from the converted files
 - `SeriesInfo`: A dictionary of DICOM metadata and conversion information for each converted image


The `SeriesInfo` value has most of the information about the converted image, including converted DICOM tags.
 - `AcqDateTime`: The acquisition date and time of the image
 - `AcquiredResolution`: The acquired, in-plane resolution of the image (list of 2 floats)
 - `AcquisitionDimension`: The number of acquisition dimensions (2D or 3D)
 - `AcquisitionMatrix`: The acquired in-plane matrix size of the image (list of 2 ints)
 - `BodyPartExamined`: The body part examined in the image
 - `ComplexImageComponent`: The complex number component represented in the image (MAGNITUDE, PHASE, REAL, IMAGINARY)
 - `ConvertImage`: Boolean indicating if the image was supposed to be converted
 - `DeviceIdentifier`: An identifier for the device used to acquire the image
 - `EPIFactor`: The echo planar imaging (EPI) factor of the image
 - `EchoTime`: The echo time (in ms) of the image
 - `EchoTrainLength`: The echo train length of the image
 - `ExContrastAgent`: Any information about the exogenous contrast agent used in the acquisition
 - `FieldOfView`: The field of view (in mm) of the image (list of 2 floats)
 - `FlipAngle`: The flip angle (in degrees) of the image
 - `ImageOrientationPatient`: The DICOM image orientation patient tag of the image (list of 6 floats)
 - `ImagePositionPatient`: The DICOM image position patient tag of the image (list of 3 floats)
 - `ImageType`: The DICOM image type tag of the image (list of strings)
 - `InstitutionName`: The institution name of the device used to acquire the image
 - `InversionTime`: The inversion time (in ms) of the image
 - `LookupName`: Any naming components for this image pulled from the lookup-table (list of strings)
 - `MagneticFieldStrength`: The magnetic field strength (in T) of the image
 - `ManualName`: Any naming components for this image pulled from the manual naming entries (list of strings)
 - `Manufacturer`: The manufacturer of the device used to acquire the image
 - `MultiFrame`: Boolean indicating if the image is a multi-frame DICOM image
 - `NiftiCreated`: Boolean indicating if the image was successfully converted to NIfTI
 - `NiftiHash`: The hash of the converted NIfTI file
 - `NiftiName`: The final filename for the converted NIfTI file.
 - `NumFiles`: Number of files (or frames) incorporated into the image (number of slices).
 - `NumberOfAverages`: The number of averages used in the acquisition
 - `PercentSampling`: The percent of k-space sampling used in the acquisition
 - `PixelBandwidth`: The pixel bandwidth (in Hz) of the image
 - `PredictedName`: Automatically generated name prediction from the DICOM metadata (list of strings)
 - `ReceiveCoilName`: The name of the receive coil used in the acquisition
 - `ReconMatrix`: The reconstructed in-plane matrix size of the image (list of 2 ints)
 - `ReconResolution`: The reconstructed, in-plane resolution of the image (list of 2 floats)
 - `RepetitionTime`: The repetition time (in ms) of the image
 - `ScanOptions`: Any scan options used in the acquisition
 - `ScannerModelName`: The model name of the scanner used to acquire the image
 - `SequenceName`: The name of the sequence used to acquire the image
 - `SequenceType`: The type of sequence used to acquire the image
 - `SequenceVariant`: The variant of the sequence used to acquire the image
 - `SeriesDescription`: The DICOM series description tag of the image
 - `SeriesNumber`: The DICOM series number tag of the image
 - `SeriesUID`: The DICOM series UID tag for the image
 - `SliceOrientation`: The slice orientation of the image (axial, sagittal, or coronal)
 - `SliceSpacing`: The slice spacing (in mm) between slices of the image
 - `SliceThickness`: The slice thickness (in mm) of the image
 - `SoftwareVersions`: The software versions of the device that acquired the image
 - `SourceHash`: The hash of the source DICOM files
 - `SourcePath`: The path to the source DICOM files (relative to session directory, e.g `dcm/...`)
 - `StudyDescription`: The study description DICOM tag for the image
 - `StudyUID`: The study UID DICOM tag for the image
 - `TriggerTime`: The trigger time (in ms) of the image (can be used to store inversion time)
 - `VariableFlipAngle`: Boolean indicating if the image used variable flip angles
